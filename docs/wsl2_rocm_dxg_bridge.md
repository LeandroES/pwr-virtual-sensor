# WSL2 ROCm Connectivity: DXG Bridge Solution for librocdxg.so

**Document type:** Architectural Decision Record — Infrastructure / Driver Integration
**Component:** PWR Digital Twin — GPU Worker Container (AMD ROCm / WSL2)
**Revision:** 1.0
**Date:** 2026-03-09

---

## 1. Executive Summary

This document records the root cause analysis and implemented solution for the ROCm GPU connectivity failure observed when running the PWR Digital Twin GPU worker container (`sensor_worker_gpu`) under Windows Subsystem for Linux 2 (WSL2) with an AMD Radeon RX 7900 XT (RDNA3). The failure manifested as a SIGSEGV (exit code 139) crash of the Celery worker process at first `import torch`, caused by an undefined symbol (`hsaKmtOpenKFD`) in the ROCm HSA runtime chain. The root cause is the absence of `librocdxg.so` — the WSL2 Direct Execution Graphics (DXG) bridge library — from the official `rocm/pytorch:latest` Docker image. The solution implemented is manual compilation of `librocdxg.so` from the upstream AMD source repository on the WSL2 host, followed by bundling the resulting binary as a build artifact in `backend/libs/` and installing it into the container image at build time via `Dockerfile.gpu`.

---

## 2. Background: The WSL2 GPU Access Model

### 2.1 Architecture Overview

WSL2 does not expose the physical GPU via the standard Linux ROCm Kernel Fusion Driver (KFD, accessed via `/dev/kfd`). Instead, Microsoft and AMD jointly defined the Direct Execution Graphics (DXG) interface: a paravirtualised kernel driver that bridges Linux user-space to the Windows AMD Adrenalin driver running on the host OS. The relevant components are:

```
Windows Host
  AMD Adrenalin Driver (≥ 23.40.27.06)
       |
  DXG Kernel Driver (dxgkrnl.sys)
       |
  ─── Hyper-V boundary ───────────────
       |
WSL2 Linux Kernel
  /dev/dxg   (DXG character device, exposed by WSL2 kernel)
       |
  /usr/lib/wsl/lib/libdxcore.so  (DXG kernel interface, volume-mounted from host)
       |
  librocdxg.so   (ROCm HSA bridge: translates hsaKmt* API to DXG ioctl calls)
       |
  libhsa-runtime64.so  (ROCm HSA runtime, calls hsaKmtOpenKFD at hsaInit())
       |
  libamdhip64.so / PyTorch HIP backend
```

This architecture requires three distinct components to co-exist within the Linux environment of the container:

1. **`/dev/dxg`** — exposed by the WSL2 kernel; passed into the container via `devices:` in `docker-compose.gpu.yml`.
2. **`libdxcore.so`** — installed by the AMD Adrenalin Windows driver into `C:\Windows\System32\lxss\lib\`, automatically volume-mounted by WSL2 at `/usr/lib/wsl/lib/`; propagated into the container via a bind mount in `docker-compose.gpu.yml`.
3. **`librocdxg.so`** — the ROCm-side bridge library; must be present somewhere on `LD_LIBRARY_PATH` inside the container.

Component 3 is the subject of this document.

### 2.2 Role of librocdxg.so

`librocdxg.so` implements the HSA Kernel Mode Thunk (KMT) API — specifically `hsaKmtOpenKFD`, `hsaKmtGetVersion`, `hsaKmtReleaseSystemProperties`, and approximately 40 additional entry points — by translating each call into the corresponding DXG ioctl call directed at `/dev/dxg`. The dynamic symbol `hsaKmtOpenKFD` is declared extern in `libhsa-runtime64.so` and is expected to be resolved at load time from `librocdxg.so`.

If the symbol is unresolved (i.e., `librocdxg.so` is absent from the linker search path), the C++ runtime raises a fatal error when `libhsa-runtime64.so` is `dlopen()`'d during `torch.cuda.is_available()` or `import torch`. Because this occurs at the C runtime level, Python's exception machinery is bypassed entirely: the process receives SIGSEGV and terminates with exit code 139, producing no Python traceback and no useful Celery error log.

---

## 3. Problem Statement

### 3.1 Missing Library in Official Image

The `rocm/pytorch:latest` Docker image (based on Ubuntu, containing ROCm 7.2.0 and PyTorch 2.10.0+rocm7.2.0) does **not** include `librocdxg.so`. Inspection of the image confirms:

```
$ find /opt/rocm /usr/lib -name "librocdxg*" 2>/dev/null
(no output)
```

This is by design: `librocdxg.so` is a WSL2-specific library that depends at build time on Windows Kernel SDK headers (`d3dkmthk.h`, `dxcore.h`, `dxcore_interface.h`, `d3dkmdt.h`) which are not part of any Linux distribution.

### 3.2 Absence from ROCm apt Repository

The `rocm/pytorch` image includes the AMD ROCm apt repository (`repo.radeon.com`). However, as of ROCm 7.2.0, no package in that repository provides `librocdxg.so`. The library is distributed exclusively as a pre-built binary bundled with the AMD Adrenalin Windows driver installation on the host, not as a standalone Linux package.

### 3.3 Host Volume Mount is Read-Only

The AMD Adrenalin driver installs `librocdxg.so` into `C:\Windows\System32\lxss\lib\`, which WSL2 volume-mounts at `/usr/lib/wsl/lib/`. In Docker under WSL2, this path can be forwarded into the container as a bind mount. The `docker-compose.gpu.yml` in this project does precisely that:

```yaml
volumes:
  - /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro
```

However, the AMD Adrenalin driver for Windows as of version 24.x does **not** install `librocdxg.so` into that path. The AMD driver for Windows ships `libd3d12.so`, `libdxcore.so`, and `libd3d12core.so` at that location, but `librocdxg.so` is a ROCm-specific component not included in the Windows driver package. Consequently, the host-mounted path does not contain the required library, and the bind mount alone is insufficient.

---

## 4. Solution: Host Compilation and Build Artifact Bundling

### 4.1 Strategy Selection

Three strategies were evaluated:

| Strategy | Feasibility | Assessment |
|---|---|---|
| Install via ROCm apt inside Dockerfile | Not feasible | Package does not exist in ROCm 7.2.0 repo |
| Build librocdxg.so inside docker build (multi-stage) | Not feasible | Requires Windows SDK headers; not available in Linux build environment |
| Build librocdxg.so on WSL2 host; bundle in Docker context | Feasible | Windows SDK headers accessible from WSL2 via /mnt/c; libdxcore.so available at runtime |
| Inject via volume mount at runtime | Partially feasible | Requires manual pre-build on each host; not reproducible across machines |

The selected strategy is option 3: compile `librocdxg.so` once on the WSL2 host (where Windows SDK headers are accessible) and commit the resulting binary to `backend/libs/`. This makes the build fully reproducible from `docker compose build` without any host-side dependencies beyond the WSL2 Docker daemon.

### 4.2 Source Repository

`librocdxg` is maintained by AMD at:

```
https://github.com/ROCm/librocdxg.git
```

The repository implements the hsaKmt API over DXG ioctls. Its CMake build system accepts a `WIN_SDK` variable pointing to the Windows SDK include directories.

### 4.3 Compilation Procedure on WSL2 Host

The following procedure was executed on the WSL2 host with Windows SDK for Windows 11 (version 10.0.26100.7705) installed at the default path `C:\Program Files (x86)\Windows Kits\10\`.

```bash
# 1. Clone the source
git clone https://github.com/ROCm/librocdxg.git /tmp/librocdxg
cd /tmp/librocdxg

# 2. Create symlinks to Windows SDK headers (avoids paths with spaces in cmake)
mkdir -p /tmp/winsdk
ln -sf "/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared" \
       /tmp/winsdk/shared
ln -sf "/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/um" \
       /tmp/winsdk/um

# 3. Configure and build
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DWIN_SDK="/tmp/winsdk/shared;/tmp/winsdk/um"
make -j$(nproc)

# 4. Verify output
ls -lh librocdxg.so.1.1.0   # expected: ~405 KB
ldd librocdxg.so.1.1.0      # must show only standard Linux libs
```

The `ldd` check confirmed that the compiled library depends only on:

- `libstdc++.so.6` (GNU C++ standard library)
- `libm.so.6` (math library)
- `libc.so.6` (C library)
- `libgcc_s.so.1` (GCC runtime)

All four are present in the `rocm/pytorch:latest` image. Critically, `libdxcore.so` is **not** a link-time dependency — it is loaded at runtime via `dlopen()` from the path specified at runtime by the HSA runtime's DXG detection logic. This means the binary is fully portable between any ROCm Linux environment.

### 4.4 Artifact Storage

The compiled binary was copied into the Docker build context:

```bash
mkdir -p /home/woz/NuclearEnergy/pwr-virtual-sensor/backend/libs/
cp /tmp/librocdxg/build/librocdxg.so.1.1.0 \
   /home/woz/NuclearEnergy/pwr-virtual-sensor/backend/libs/
```

Soname symlinks are created at image build time (see Section 4.5) rather than committed to the repository, to keep the repository clean.

### 4.5 Dockerfile Integration

The following block was added to `backend/Dockerfile.gpu` immediately after `WORKDIR /app`, before the virtualenv configuration:

```dockerfile
# ── Install librocdxg (WSL2 DXG bridge for ROCm HSA runtime) ─────────────────
# librocdxg.so is NOT bundled in rocm/pytorch:latest and is absent from the
# ROCm apt repo included in the image.  We pre-compile it on the WSL2 host
# (where Windows SDK headers + libdxcore.so are available) and ship the binary
# directly in the Docker context under backend/libs/.
#
# At runtime it is loaded by libhsa-runtime64.so when HSA_ENABLE_DXG_DETECTION=1
# and /dev/dxg is present.  libdxcore.so (the underlying Windows DXG kernel
# interface) is resolved dynamically via dlopen() from /usr/lib/wsl/lib.
COPY libs/librocdxg.so.1.1.0 /opt/rocm/lib/
RUN ln -sf /opt/rocm/lib/librocdxg.so.1.1.0 /opt/rocm/lib/librocdxg.so.1 && \
    ln -sf /opt/rocm/lib/librocdxg.so.1      /opt/rocm/lib/librocdxg.so  && \
    ldconfig
```

Installing into `/opt/rocm/lib/` ensures the library is:

1. On the `LD_LIBRARY_PATH` configured in both the Dockerfile and `docker-compose.gpu.yml`.
2. In the `ldconfig` cache, allowing the dynamic linker to resolve it without specifying a full path.
3. Distinct from the host-mounted `/usr/lib/wsl/lib/` (which is read-only and does not contain this file), avoiding any conflict.

---

## 5. GPU Detection Logic

### 5.1 Two-Layer Protection Model

A two-layer detection and protection scheme was implemented to prevent the SIGSEGV in any deployment configuration:

**Layer 1 — Shell entrypoint (`docker-entrypoint-gpu.sh`)**

Executes before Celery starts. Searches for `librocdxg.so` in a list of candidate directories:

```bash
DXG_LIB=""
for _libdir in /usr/lib/wsl/lib /opt/rocm/lib /usr/local/lib; do
    if [ -f "${_libdir}/librocdxg.so" ]; then
        DXG_LIB="${_libdir}/librocdxg.so"
        break
    fi
done

if [ -c "${DXG_DEVICE}" ] && [ -n "${DXG_LIB}" ]; then
    GPU_MODE="wsl2-dxg"
fi
```

If neither WSL2 DXG nor Linux KFD is detected, the entrypoint exports `HIP_VISIBLE_DEVICES=-1` before invoking `celery`. This prevents `hipGetDeviceCount()` from calling `hsaInit()`, which is the function that dereferences `hsaKmtOpenKFD` and would trigger the crash.

**Layer 2 — Celery worker_ready signal (`app/worker/celery_app.py`)**

Fires inside the worker process after Celery initialises. Repeats the same detection logic before any `import torch`:

```python
_WSL2_DXG_LIB_CANDIDATES: list[str] = [
    "/usr/lib/wsl/lib/librocdxg.so",
    "/opt/rocm/lib/librocdxg.so",
    "/usr/local/lib/librocdxg.so",
]

def _find_rocdxg_lib() -> str | None:
    return next((p for p in _WSL2_DXG_LIB_CANDIDATES if os.path.exists(p)), None)

def _gpu_access_available() -> bool:
    wsl2_ok  = os.path.exists(_WSL2_DXG_DEVICE) and _find_rocdxg_lib() is not None
    linux_ok = os.path.exists(_LINUX_KFD)
    return wsl2_ok or linux_ok
```

Layer 2 handles the case where the entrypoint's environment variable is overridden by a subsequent `docker-compose` `environment:` key, or where the GPU became inaccessible after entrypoint execution (e.g., driver reset on WSL2).

### 5.2 Environment Variables

The following environment variables are required for GPU operation on WSL2 and are set in `docker-compose.gpu.yml`:

| Variable | Value | Purpose |
|---|---|---|
| `HSA_ENABLE_DXG_DETECTION` | `1` | Instructs HSA runtime to use `/dev/dxg` for GPU enumeration |
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | Forces HIP to treat the GPU as RDNA3 (gfx1100); required when the device string is absent from the ROCm LLVM target table |
| `LD_LIBRARY_PATH` | `/usr/lib/wsl/lib:/opt/rocm/lib:...` | Ensures `libdxcore.so` (host-mounted) and `librocdxg.so` (image-installed) are both resolvable |
| `AMD_SERIALIZE_KERNEL` | `3` | Serialises all GPU kernels; prevents race conditions in the DXG translation layer under WSL2 |

---

## 6. Runtime Verification

### 6.1 Expected Worker Log on Successful GPU Initialisation

After a successful build and deployment, the worker log should contain:

```
[entrypoint] GPU mode: WSL2 DXG bridge
[entrypoint]   /dev/dxg       : found
[entrypoint]   librocdxg.so   : /opt/rocm/lib/librocdxg.so
[entrypoint] Handing off to: celery -A app.worker.celery_app worker ...

[GPU WARMUP] GPU access interface detected:
  WSL2  — /dev/dxg: found
          librocdxg.so: /opt/rocm/lib/librocdxg.so
  HSA_OVERRIDE_GFX_VERSION = 11.0.0
  HSA_ENABLE_DXG_DETECTION = 1
[GPU WARMUP] Device : AMD Radeon RX 7900 XT | VRAM: 20.00 GiB | CUs: 84
[GPU WARMUP] Ready — VRAM allocated: 28.3 MB | reserved: 30.0 MB
```

### 6.2 Smoke Test

```bash
docker exec sensor_worker_gpu python3 -c \
  "import torch; print('GPU:', torch.cuda.get_device_name(0))"
# Expected: GPU: AMD Radeon RX 7900 XT
```

### 6.3 Health Check

The `healthcheck` in `docker-compose.gpu.yml` verifies GPU availability every 60 seconds:

```yaml
healthcheck:
  test:
    - "CMD"
    - "python"
    - "-c"
    - >-
      import torch, sys;
      ok = torch.cuda.is_available();
      t = torch.ones(16, device='cuda') if ok else None;
      print('GPU OK:', torch.cuda.get_device_name(0) if ok else 'CPU-ONLY');
      sys.exit(0 if ok else 1)
  interval: 60s
  timeout: 30s
  retries: 3
  start_period: 90s
```

The `start_period` of 90 seconds accommodates the GPU warmup routine in `celery_app.py`, which executes a 1024x1024 SGEMM on first startup to pre-warm the ROCm HIP kernel cache (first-run JIT compilation: 5-15 seconds).

---

## 7. Reproducibility and Maintenance

### 7.1 Re-compilation Trigger Conditions

The pre-compiled `librocdxg.so.1.1.0` artifact in `backend/libs/` must be recompiled when:

- The upstream `ROCm/librocdxg` repository releases a new version with API changes.
- The ROCm runtime version in `rocm/pytorch:latest` advances to a version that requires a different hsaKmt ABI.
- The target Linux distribution in the base image changes its `libstdc++` ABI in a backwards-incompatible way.

The compilation procedure in Section 4.3 is fully documented and repeatable on any WSL2 host with AMD Adrenalin driver and Windows SDK for Windows 11 installed.

### 7.2 AMD Adrenalin Driver Version Requirement

GPU access via the DXG bridge requires AMD Adrenalin driver version 23.40.27.06 or later on the Windows host. This version introduced WSL2 GPU passthrough support. The driver is available at the AMD support site.

### 7.3 Windows SDK Version

The compilation was performed against Windows SDK 10.0.26100.7705 (Windows 11 24H2). The required headers (`d3dkmthk.h`, `dxcore.h`, `dxcore_interface.h`) are stable across SDK releases targeting Windows 10 20H1 and later; re-compilation against newer SDK versions is not required unless the DXG ioctl interface changes.

---

## 8. References

1. Microsoft, "GPU compute support in WSL," Windows Subsystem for Linux Documentation, https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute, 2024.
2. AMD, "ROCm on Windows Subsystem for Linux," AMD ROCm Documentation, https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install-rocm.html, 2024.
3. AMD ROCm, "librocdxg — ROCm DXG Bridge Library," GitHub, https://github.com/ROCm/librocdxg, 2024.
4. AMD, "HSA Platform System Architecture Specification," HSA Foundation, Version 1.2, 2019.
5. Microsoft, "DirectX Graphics Kernel (dxgkrnl)," Windows Driver Documentation, 2023.
