"""
Celery application factory and GPU warmup for the PWR Digital Twin worker.

GPU crash prevention (exit 139 / SIGSEGV)
-----------------------------------------
The ROCm HSA runtime (libhsa-runtime64.so) calls hsaKmtOpenKFD when it
initialises.  This symbol is provided by:

  WSL2 path   librocdxg.so  (DXG bridge, installed by AMD Adrenalin driver)
  Linux path  KFD kernel module  (exposed as /dev/kfd)

If neither is available, hsaKmtOpenKFD is an undefined symbol.  The dynamic
linker finds it undefined, the C++ runtime aborts, and the process exits with
signal 11 (SIGSEGV) / exit code 139.  Python cannot catch this; there is no
traceback — the process just disappears.

Two-layer protection implemented here:

  Layer 1 — docker-entrypoint-gpu.sh (shell, runs BEFORE Celery starts):
    Checks for /dev/dxg + librocdxg.so (WSL2) or /dev/kfd (Linux).
    If neither found: exports HIP_VISIBLE_DEVICES=-1 before exec-ing Celery.
    HIP_VISIBLE_DEVICES=-1 makes hipGetDeviceCount() return 0 without ever
    calling hsaInit(), so the HSA runtime is never loaded.

  Layer 2 — _warmup_gpu_cache (this file, fires via worker_ready signal):
    Repeats the same check inside Python before calling `import torch`.
    Acts as a safety net in case the entrypoint variable is overridden.
    Also handles the case where the GPU was present at startup but later
    becomes inaccessible (e.g., driver reset on WSL2).

GPU warmup (when GPU IS available)
-----------------------------------
The ROCm runtime loads its shared libraries and compiles HIP kernels lazily.
The FIRST call to any GPU operation in a fresh process triggers:
  1. ROCm runtime initialisation            (~1-3 s)
  2. LLVM-based HIP kernel JIT compilation  (~5-15 s, cached after first run)
  3. rocBLAS library loading                (~0.5-1 s)

If this happens inside the first run_virtual_sensor_job task, the total
startup latency (~20 s) looks like a freeze and can hit soft_time_limit.
The warmup function runs a 1024×1024 SGEMM to trigger all of this at worker
startup, so every subsequent task starts with the GPU fully ready.
"""

from __future__ import annotations

import logging
import os
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
    worker_prefetch_multiplier=1,
    result_expires=86_400,
)

# ── GPU warmup ────────────────────────────────────────────────────────────────

logger = get_task_logger(__name__)

_WARMUP_N: int            = 1_024   # SGEMM dimension — exercises rocBLAS path
_WARMUP_ENSEMBLE_N: int   = 1_000   # prime allocator for EnsembleSolver shapes
_WARMUP_STATE_DIM: int    = 9       # state vector dimension

# GPU access interface paths
_WSL2_DXG_DEVICE: str = "/dev/dxg"
_WSL2_DXG_LIB:    str = "/usr/lib/wsl/lib/librocdxg.so"
_LINUX_KFD:        str = "/dev/kfd"


def _gpu_access_available() -> bool:
    """
    Return True if a ROCm-compatible GPU access interface is present.

    Checks in order:
      1. WSL2 DXG bridge: /dev/dxg character device AND librocdxg.so library
      2. Native Linux KFD: /dev/kfd character device
    """
    wsl2_ok  = os.path.exists(_WSL2_DXG_DEVICE) and os.path.exists(_WSL2_DXG_LIB)
    linux_ok = os.path.exists(_LINUX_KFD)
    return wsl2_ok or linux_ok


@worker_ready.connect
def _warmup_gpu_cache(sender: Any, **kwargs: Any) -> None:
    """
    Pre-warm the GPU context once at worker startup (via worker_ready signal).

    Layer 2 protection: checks GPU hardware access before importing torch.
    If the ROCm DXG bridge or KFD is unavailable, disables HIP device
    enumeration and returns immediately — preventing the SIGSEGV that would
    result from the HSA runtime trying to call the missing hsaKmtOpenKFD.

    When GPU IS available, runs a (1024×1024) SGEMM to:
      • Force CUDA/ROCm context creation.
      • Trigger HIP kernel JIT compilation for SGEMM (cached for next runs).
      • Load rocBLAS.
      • Pre-warm the caching allocator for EnsembleSolver tensor shapes.
    """
    # ── Layer 2: hardware pre-check ─────────────────────────────────────────
    if not _gpu_access_available():
        logger.info(
            "[GPU WARMUP] No GPU access interface detected:\n"
            "  WSL2  — %s: %s\n"
            "          %s: %s\n"
            "  Linux — %s: %s\n"
            "Disabling HIP device enumeration (HIP_VISIBLE_DEVICES=-1) "
            "to prevent ROCm HSA runtime segfault. Worker → CPU-only mode.",
            _WSL2_DXG_DEVICE, "found" if os.path.exists(_WSL2_DXG_DEVICE) else "NOT FOUND",
            _WSL2_DXG_LIB,    "found" if os.path.exists(_WSL2_DXG_LIB)    else "NOT FOUND",
            _LINUX_KFD,       "found" if os.path.exists(_LINUX_KFD)       else "NOT FOUND",
        )
        os.environ.setdefault("HIP_VISIBLE_DEVICES",  "-1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        return

    # ── GPU access confirmed — proceed with torch warmup ────────────────────
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            logger.info(
                "[GPU WARMUP] torch.cuda.is_available() = False "
                "(GPU access interface present but HIP returned no devices). "
                "Check HSA_OVERRIDE_GFX_VERSION and device passthrough config."
            )
            return

        device     = torch.device("cuda:0")
        props      = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / (1024 ** 3)

        logger.info(
            "[GPU WARMUP] Device : %s | VRAM: %.2f GiB | CUs: %d",
            props.name, vram_total, props.multi_processor_count,
        )

        # SGEMM — forces rocBLAS / cuBLAS init + HIP JIT compilation
        a   = torch.randn(_WARMUP_N, _WARMUP_N, dtype=torch.float32, device=device)
        b   = torch.randn(_WARMUP_N, _WARMUP_N, dtype=torch.float32, device=device)
        _c  = torch.mm(a, b)
        torch.cuda.synchronize()
        del a, b, _c

        # Prime the caching allocator for EnsembleSolver (N, 9) shapes
        _dummy = torch.empty(
            _WARMUP_ENSEMBLE_N, _WARMUP_STATE_DIM,
            dtype=torch.float32, device=device,
        )
        torch.cuda.synchronize()
        del _dummy

        alloc_mb    = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(0)  / (1024 ** 2)

        logger.info(
            "[GPU WARMUP] Ready — VRAM allocated: %.1f MB | reserved: %.1f MB",
            alloc_mb, reserved_mb,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[GPU WARMUP] Warmup failed: %s — GPU will be initialised lazily "
            "on the first task (expect 5-20 s startup latency).",
            exc,
        )
