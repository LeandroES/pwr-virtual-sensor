"""
Celery application factory and GPU warmup for the PWR Digital Twin worker.

GPU warmup strategy
-------------------
PyTorch / ROCm loads its shared libraries and compiles HIP kernels lazily —
the FIRST call to any GPU operation in a fresh process triggers:
  1. ROCm runtime initialization            (~1–3 s)
  2. LLVM-based HIP kernel JIT compilation  (~5–15 s on first launch,
                                             cached on subsequent launches)
  3. cuBLAS / rocBLAS library loading       (~0.5–1 s)

Total first-operation latency: up to ~20 s.

If this happens inside the first ``run_virtual_sensor_job`` task, the Celery
soft_time_limit may fire before the actual computation begins, the memory
allocation spike appears as an OOM to the OS, and the user sees a mysterious
5-20 second freeze on the API side.

The ``_warmup_gpu_cache`` function below hooks into the ``worker_ready``
Celery signal — it fires once, immediately after the worker process completes
its startup sequence (broker connection, task discovery).  It runs a small
(1 024 × 1 024) matrix multiply to:
  • Force CUDA/ROCm context creation.
  • Trigger HIP kernel JIT compilation for SGEMM.
  • Pre-allocate the PyTorch CUDA caching allocator's memory pool.
  • Load rocBLAS / cuBLAS.

Subsequent task executions skip all of this overhead; the GPU is immediately
ready.  Total warmup VRAM cost: ≈ 8 MB (released immediately after warmup).
"""

from __future__ import annotations

import logging
from typing import Any

from celery import Celery
from celery.signals import worker_ready
from celery.utils.log import get_task_logger

from app.core.config import settings

# ── Celery application ────────────────────────────────────────────────────────

celery_app = Celery(
    "pwr_twin",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Disable prefetch so long GPU tasks don't hog the worker slot while
    # another shorter task is waiting in the queue.
    worker_prefetch_multiplier=1,
    # Keep task results in the backend for 24 h (enough for the frontend
    # to poll) then let Redis evict them automatically.
    result_expires=86_400,
)

# ── GPU warmup ────────────────────────────────────────────────────────────────

logger = get_task_logger(__name__)

# Size of the dummy SGEMM used to force rocBLAS / cuBLAS library loading.
# 1 024 × 1 024 fp32 matmul uses ~8 MB total; large enough to exercise the
# BLAS path, small enough to not fragment the caching allocator.
_WARMUP_N: int = 1_024

# A small ensemble-sized allocation (N=1 000, state dim=9) is created and
# freed after the matmul to pre-warm the caching allocator's free-block list
# for the typical shapes used by EnsembleSolver.
_WARMUP_ENSEMBLE_N: int = 1_000
_WARMUP_STATE_DIM: int  = 9


@worker_ready.connect
def _warmup_gpu_cache(sender: Any, **kwargs: Any) -> None:
    """
    Pre-warm the PyTorch GPU context at Celery worker startup.

    This function runs ONCE per worker process, triggered by the
    ``worker_ready`` signal (fired after broker connection + task discovery
    complete, before the worker starts accepting tasks).

    Steps
    -----
    1. Check torch.cuda.is_available().
       - Not available → log INFO and return immediately (CPU-only mode).
    2. Print GPU model and VRAM capacity.
    3. Allocate two (WARMUP_N, WARMUP_N) fp32 tensors on the device.
    4. Run torch.mm() → forces rocBLAS/cuBLAS library init and HIP JIT.
    5. torch.cuda.synchronize() → block until the op completes on-device.
    6. Free the matmul tensors.
    7. Allocate and free a (WARMUP_ENSEMBLE_N, WARMUP_STATE_DIM) tensor
       to prime the caching allocator's free-list for EnKF workloads.
    8. Log post-warmup VRAM usage.

    Any exception is caught and logged as a WARNING — the worker continues
    normally; the GPU will be initialised lazily on the first real task
    (with the latency penalty described in the module docstring).
    """
    try:
        import torch  # noqa: PLC0415

        # ── 1. Availability check ─────────────────────────────────────────
        if not torch.cuda.is_available():
            logger.info(
                "[GPU WARMUP] torch.cuda.is_available() = False.  "
                "Worker running in CPU-only mode.  "
                "Virtual sensor jobs will use device='cpu' (slower)."
            )
            return

        device = torch.device("cuda:0")

        # ── 2. Device info ───────────────────────────────────────────────
        props       = torch.cuda.get_device_properties(0)
        device_name = props.name
        vram_total  = props.total_memory / (1024 ** 3)   # bytes → GiB
        sm_count    = props.multi_processor_count

        logger.info(
            "[GPU WARMUP] Device: %s | VRAM: %.2f GiB | CUs: %d",
            device_name, vram_total, sm_count,
        )

        # ── 3–4. SGEMM — forces rocBLAS / cuBLAS + HIP JIT ───────────────
        # torch.randn allocates + fills the tensor; torch.mm exercises the
        # full BLAS dispatch path including kernel selection and launch.
        a = torch.randn(_WARMUP_N, _WARMUP_N, dtype=torch.float32, device=device)
        b = torch.randn(_WARMUP_N, _WARMUP_N, dtype=torch.float32, device=device)
        _c = torch.mm(a, b)

        # ── 5. Synchronize — ensure the op completed before we continue ───
        torch.cuda.synchronize()

        # ── 6. Free matmul tensors ────────────────────────────────────────
        del a, b, _c

        # ── 7. Prime the caching allocator for EnsembleSolver shapes ─────
        # EnsembleSolver pre-allocates (N, 9) tensors.  Touching a tensor
        # of that shape now ensures the allocator has a free block of the
        # right size when the first virtual-sensor task runs.
        _dummy_state = torch.empty(
            _WARMUP_ENSEMBLE_N, _WARMUP_STATE_DIM,
            dtype=torch.float32, device=device,
        )
        torch.cuda.synchronize()
        del _dummy_state

        # ── 8. Post-warmup diagnostics ────────────────────────────────────
        # memory_allocated: bytes currently in use by live tensors.
        # memory_reserved: bytes held by the caching allocator (may be > 0
        #   even after del, because the allocator retains freed blocks).
        allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved_mb  = torch.cuda.memory_reserved(0)  / (1024 ** 2)

        logger.info(
            "[GPU WARMUP] Context ready — "
            "allocated: %.1f MB | reserved (cache): %.1f MB.  "
            "Worker accepting tasks.",
            allocated_mb, reserved_mb,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[GPU WARMUP] Warmup failed: %s.  "
            "The worker will initialise the GPU on the first task "
            "(expect 5–20 s latency on that job).",
            exc,
        )
