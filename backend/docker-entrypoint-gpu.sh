#!/bin/bash
# docker-entrypoint-gpu.sh
#
# GPU availability pre-check — runs before Celery starts.
#
# WHY THIS EXISTS
# ──────────────────────────────────────────────────────────────────────────────
# The ROCm HSA runtime (libhsa-runtime64.so) crashes with SIGSEGV (exit 139)
# when torch is imported and neither of the following is available:
#
#   • WSL2 path  — librocdxg.so  (DXG bridge library, installed by AMD's
#                                  Windows driver) + /dev/dxg device
#   • Linux path — /dev/kfd      (ROCm Kernel Fusion Driver, requires the
#                                  amdgpu + amdkfd kernel modules to be loaded)
#
# The crash happens at C++ runtime level; Python cannot catch a SIGSEGV.
# Setting HIP_VISIBLE_DEVICES=-1 BEFORE Celery starts prevents the HIP layer
# from ever calling hsaInit(), which is what triggers the crash.
#
# ENVIRONMENT DETECTION
# ──────────────────────────────────────────────────────────────────────────────
# WSL2:   /dev/dxg exists AND /usr/lib/wsl/lib/librocdxg.so is present
#         → The AMD Adrenalin Windows driver ≥ 23.40.27.06 installs
#           librocdxg.so into /usr/lib/wsl/lib/ via the WSL GPU driver bridge.
#         → The volume mount in docker-compose.gpu.yml exposes this path.
#
# Linux:  /dev/kfd exists
#         → ROCm KFD kernel module is loaded; GPU is accessible natively.
#
# Neither: GPU is not accessible. We set HIP_VISIBLE_DEVICES=-1 so that
#         torch.cuda.is_available() returns False without touching the HSA
#         runtime, and the worker runs in CPU-only mode.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

GPU_MODE="none"

# ── WSL2 DXG bridge check ──────────────────────────────────────────────────────
DXG_DEVICE="/dev/dxg"
DXG_LIB="/usr/lib/wsl/lib/librocdxg.so"

if [ -c "${DXG_DEVICE}" ] && [ -f "${DXG_LIB}" ]; then
    GPU_MODE="wsl2-dxg"
fi

# ── Native Linux KFD check ────────────────────────────────────────────────────
if [ -c "/dev/kfd" ]; then
    GPU_MODE="linux-kfd"
fi

# ── Act on detection result ───────────────────────────────────────────────────
case "${GPU_MODE}" in

  "wsl2-dxg")
    echo "[entrypoint] GPU mode: WSL2 DXG bridge"
    echo "[entrypoint]   /dev/dxg       : found"
    echo "[entrypoint]   librocdxg.so   : ${DXG_LIB}"
    ;;

  "linux-kfd")
    echo "[entrypoint] GPU mode: native Linux ROCm (KFD)"
    echo "[entrypoint]   /dev/kfd       : found"
    ;;

  "none")
    echo "[entrypoint] ──────────────────────────────────────────────────────"
    echo "[entrypoint] WARNING: No GPU access interface found."
    echo "[entrypoint]"
    echo "[entrypoint]   WSL2 DXG  — /dev/dxg not present OR"
    echo "[entrypoint]               librocdxg.so missing at:"
    echo "[entrypoint]               ${DXG_LIB}"
    echo "[entrypoint]"
    echo "[entrypoint]   Linux KFD — /dev/kfd not present"
    echo "[entrypoint]"
    echo "[entrypoint] Disabling HIP device enumeration to prevent ROCm HSA"
    echo "[entrypoint] runtime segfault (exit 139). Worker → CPU-only mode."
    echo "[entrypoint]"
    echo "[entrypoint] To enable GPU support:"
    echo "[entrypoint]   WSL2: Install AMD Adrenalin driver ≥ 23.40.27.06"
    echo "[entrypoint]         on the Windows host and restart WSL2."
    echo "[entrypoint]   Linux: Load amdgpu + amdkfd kernel modules and"
    echo "[entrypoint]          add user to 'render' and 'video' groups."
    echo "[entrypoint] ──────────────────────────────────────────────────────"

    # HIP_VISIBLE_DEVICES=-1 makes hipGetDeviceCount() return 0 without
    # calling hsaInit(). This prevents the SIGSEGV that occurs when
    # hsaKmtOpenKFD (normally provided by librocdxg.so or KFD) is missing.
    export HIP_VISIBLE_DEVICES="-1"
    export CUDA_VISIBLE_DEVICES="-1"

    # Disable DXG detection so the HSA runtime doesn't attempt to dlopen
    # librocdxg.so, which would log spurious errors even when HIP is disabled.
    export HSA_ENABLE_DXG_DETECTION="0"
    ;;

esac

echo "[entrypoint] Handing off to: $*"
exec "$@"
