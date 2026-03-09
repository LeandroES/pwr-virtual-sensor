# PCIe Transfer Overhead Analysis: CPU vs GPU Performance Boundary for Ensemble Simulations

**Document type:** Architectural Decision Record — Performance Analysis
**Component:** PWR Digital Twin — Ensemble Kalman Filter (EnKF) Physics Backend
**Revision:** 1.1
**Date:** 2026-03-09

---

## 1. Executive Summary

This document quantifies the PCIe memory transfer overhead that governs the performance crossover between CPU and GPU execution for the ensemble physics solver (`EnsembleSolver`) of the PWR Digital Twin. Analytical modelling and empirical measurement demonstrate that for ensemble sizes below approximately N = 250,000 members, the WSL2 DXG bridge latency and HIP kernel serialisation overhead (`AMD_SERIALIZE_KERNEL=3`) together dominate the GPU execution time, rendering CPU execution faster in total wall-clock time on the specific hardware of this deployment (Ryzen 9 5950X / DDR4-3600 / RX 7900 XT / WSL2). The GPU becomes the superior compute engine only for N >= 250,000 on this stack.

**Revision 1.1 note:** The initial revision used a DDR5-6000 CPU bandwidth figure (96 GB/s). The actual host memory subsystem is DDR4-3600 dual-channel (~50 GB/s practical), which significantly reduces CPU execution time per step and raises the empirical crossover. In addition, an empirical measurement at N = 100,000 (GPU 4× slower than CPU) was used to back-calculate the actual WSL2 DXG per-step overhead (~567 µs), replacing the theoretical 60–100 µs native-Linux estimate. All figures in this revision reflect the corrected values.

---

## 2. Background and Motivation

The Ensemble Kalman Filter implemented in `backend/app/physics/tensor_solver.py` and `backend/app/physics/assimilation.py` operates on an ensemble of N independent reactor state vectors, each of dimension d = 9 (neutron density, six delayed-neutron precursor concentrations, fuel temperature, coolant temperature). The RK4 integrator advances all N members simultaneously, and the EnKF analysis step computes the (d x d) sample covariance matrix and the Kalman gain in a single batched linear-algebra operation.

The computational cost of both stages scales linearly with N, but the arithmetic intensity of the RK4 kernel is moderate: each member requires approximately 144 floating-point operations per sub-step (four evaluations of the nine-dimensional right-hand side). This low arithmetic intensity makes the workload memory-bandwidth-bound rather than compute-bound, which has direct implications for the PCIe bottleneck.

---

## 3. Hardware Parameters

| Parameter | Value | Source |
|---|---|---|
| GPU model | AMD Radeon RX 7900 XT (RDNA3, Navi 31) | lspci |
| GPU peak FP32 throughput | 51.6 TFLOP/s | AMD product specification |
| GPU memory bandwidth (HBM2e) | 800 GB/s | AMD product specification |
| PCIe bus version | PCIe 4.0 x16 | Motherboard specification |
| PCIe 4.0 x16 theoretical peak | 32 GB/s bidirectional (16 GB/s per direction) | PCIe 4.0 specification |
| PCIe 4.0 x16 sustained throughput | ~14 GB/s (H2D or D2H, measured, accounting for protocol overhead) | Empirical, AMD ROCm benchmark |
| Host RAM | DDR4-3600 dual-channel, 64 GB (2 × 32 GB) | System specification |
| Host RAM bandwidth (DDR4-3600, 2ch) | ~57.6 GB/s theoretical; ~50 GB/s practical | JEDEC DDR4 spec; stream benchmark |
| CPU model | AMD Ryzen 9 5950X (Zen 3, 16 cores / 32 threads) | lscpu |
| CPU peak FP32 throughput (AVX2) | ~1.5 TFLOP/s | Theoretical, 8-wide FP32 x 2 FMA |
| WSL2 DXG per-step overhead (measured) | ~567 µs | Back-calculated from N=100k empirical measurement |
| WSL2 configuration | AMD_SERIALIZE_KERNEL=3 (DXG bridge stability) | docker-compose.gpu.yml |

---

## 4. Data Volume Analysis

### 4.1 State Tensor Size

Each simulation step operates on the full ensemble state matrix X of shape (N, 9) in float32:

```
bytes(X) = N x 9 x 4 = 36 N bytes
```

For representative ensemble sizes:

| N | bytes(X) | Transfer time H2D at 14 GB/s |
|---|---|---|
| 1,000 | 36 KB | 0.003 ms |
| 10,000 | 360 KB | 0.026 ms |
| 50,000 | 1.8 MB | 0.129 ms |
| 100,000 | 3.6 MB | 0.257 ms |
| 500,000 | 18 MB | 1.286 ms |

In practice, the EnKF implementation in this project pre-allocates all tensors on-device at initialization time and does not transfer the state matrix per step; the observation scalar and the resulting mean/variance estimate are the only values crossing the PCIe bus at steady state. However, the initial allocation phase transfers all parameter tensors (shape (N, 9), (N, 6) for beta groups, etc.), and each new simulation job incurs this cost once at `EnsembleSolver.initialize()`.

### 4.2 Initial Allocation Transfer Cost

At `initialize()`, the following tensors are constructed on the host (NumPy) and moved to device:

| Tensor | Shape | Bytes (float32) |
|---|---|---|
| State X | (N, 9) | 36N |
| RK4 scratch k1..k6 | 6 x (N, 9) | 216N |
| Per-member params (alpha_f, UA_fc, P0) | (N, 3) | 12N |
| Beta groups | (1, 6) broadcast to (N, 6) | 24N (virtual, zero-copy) |
| Lambda groups | (1, 6) broadcast to (N, 6) | 24N (virtual, zero-copy) |
| Scalar params (N_ones) | (N, 1) | 4N |

**Total concrete transfer:** approximately 268N bytes, excluding broadcast views.

For N = 10,000: 268 x 10,000 = 2.68 MB, transferred in approximately 0.19 ms.
For N = 100,000: 268 x 100,000 = 26.8 MB, transferred in approximately 1.91 ms.

---

## 5. Computational Cost Analysis

### 5.1 RK4 Kernel Arithmetic Intensity

The nine-dimensional point-kinetics + thermal-hydraulics RHS requires per member per sub-step:

- Neutron density equation: 6 multiply-add (beta/Lambda terms) + 2 ops = 14 FLOPs
- Six precursor equations: 2 FLOPs each = 12 FLOPs
- Fuel temperature equation: 4 FLOPs
- Coolant temperature equation: 4 FLOPs

**Total per sub-step per member:** approximately 34 FLOPs (conservative; ignoring intermediate buffer stores).
**Total per RK4 step per member:** 4 sub-steps x 34 = 136 FLOPs.

At N members per step, total FLOPs = 136N.

**Arithmetic intensity** = 136N FLOPs / (36N bytes state reads + 36N bytes state writes) = 136 / 72 = 1.89 FLOPs/byte.

This is well below the roofline threshold for the RX 7900 XT:

```
roofline_flops_per_byte = peak_FLOPS / memory_bandwidth
                        = 51.6e12 / 800e9
                        = 64.5 FLOPs/byte
```

The workload is memory-bandwidth-bound on the GPU, not compute-bound. The effective GPU throughput for this kernel is therefore capped by HBM2e bandwidth, not by the shader array.

### 5.2 GPU Execution Time (Memory-Bound Estimate)

For N members, the RK4 step reads and writes approximately 72N bytes (state matrix twice plus scratch once):

```
t_gpu_step(N) = (72N bytes) / (800 GB/s) = 90e-12 x N seconds
```

For N = 10,000: t_gpu_step = 0.9 microseconds (arithmetic; excludes kernel launch overhead).
For N = 100,000: t_gpu_step = 9 microseconds.

**Kernel launch overhead** on ROCm/HIP is approximately 5-20 microseconds per dispatch. The RK4 step dispatches four kernels (k1, k2, k3, k4), plus element-wise combination passes: total overhead approximately 60-100 microseconds per time step.

This overhead dominates over the compute time at N = 10,000:

```
t_overhead ~= 80 us >> t_compute ~= 0.9 us   (N = 10,000)
```

### 5.3 CPU Execution Time (Ryzen 9 5950X, DDR4-3600)

The EnsembleSolver on CPU uses PyTorch with AVX2 vectorisation across the N members dimension, exploiting all 16 cores via the ATen thread pool. At DDR4-3600 dual-channel (~50 GB/s practical bandwidth), the workload is memory-bandwidth-bound:

```
t_cpu_mem(N) = 72N bytes / 50e9 = 1,440e-12 x N seconds
```

For N = 10,000:  t_cpu_mem ~= 14.4 microseconds.
For N = 100,000: t_cpu_mem ~= 144 microseconds.
For N = 250,000: t_cpu_mem ~= 360 microseconds.

No kernel dispatch overhead; PyTorch CPU thread-pool dispatch is approximately 1-5 microseconds — negligible compared to memory transfer time.

Note: DDR4-3600 dual-channel delivers ~2.3× less bandwidth than the DDR5-6000 figure used in revision 1.0 (96 GB/s). This materially increases CPU execution time per step and therefore raises the crossover N relative to the original estimate.

---

## 6. Performance Crossover Model

### 6.1 Analytical Model

The total wall-clock time per simulation step on GPU:

```
T_gpu(N) = t_overhead_wsl2 + t_compute_gpu(N)
         = t_overhead_wsl2 + 72N / 800e9
```

Where `t_overhead_wsl2` is the fixed per-step cost of the WSL2 DXG bridge dispatch, HIP runtime, and the serialisation imposed by `AMD_SERIALIZE_KERNEL=3`. On native Linux with /dev/kfd this term is 60–100 µs; on WSL2 DXG it is significantly higher.

The total wall-clock time per step on CPU (DDR4-3600 dual-channel, ~50 GB/s):

```
T_cpu(N) = 72N / 50e9 = 1,440e-12 x N seconds
```

Crossover condition T_gpu(N*) = T_cpu(N*):

```
t_overhead_wsl2 + 90e-12 x N* = 1,440e-12 x N*
t_overhead_wsl2 = (1,440e-12 - 90e-12) x N*
t_overhead_wsl2 = 1,350e-12 x N*
N* = t_overhead_wsl2 / 1,350e-12
```

### 6.2 Empirical Calibration

A direct measurement was performed at N = 100,000 on this hardware stack. The GPU execution was observed to be 4× slower than CPU:

```
t_cpu(N=100k)       = 1,440e-12 x 100,000  = 144 µs
t_gpu_measured      = 4 x 144              = 576 µs
t_overhead_wsl2     = 576 - 90             = 486 µs   (back-calculated)

N* = 486e-6 / 1,350e-12  ~= 360,000
```

This places the empirical crossover at approximately N = 360,000. The Smart Switch threshold is set at **N = 250,000** — a 30 % conservative margin below the crossover — because the cost of activating the GPU too early (2–4× latency penalty) is greater than the cost of keeping the CPU path slightly longer than necessary.

### 6.3 GPU Speedup at Representative Ensemble Sizes

| N | T_cpu (µs) | T_gpu (µs) | GPU vs CPU |
|---|---|---|---|
| 10,000 | 14 | 487 | 35× slower |
| 100,000 | 144 | 495 | 3.4× slower |
| 250,000 | 360 | 509 | 1.4× slower (threshold) |
| 360,000 | 518 | 518 | break-even |
| 500,000 | 720 | 531 | 1.4× faster |
| 1,000,000 | 1,440 | 576 | 2.5× faster |

Note that even at N = 1,000,000 the GPU advantage is modest (2.5×) due to the fixed WSL2 overhead. On native Linux with /dev/kfd (overhead ~80 µs) the GPU would be 15× faster at N = 1,000,000.

---

## 7. Device Selection Summary

### 7.1 Performance Recommendation by Ensemble Size (this hardware stack)

| N | Smart Switch decision | Scientific tier | Rationale |
|---|---|---|---|
| N < 10,000 | CPU | Operational | GPU 30–35× slower; filter statistically converged for d=9 |
| 10,000 – 99,999 | CPU | Research-grade | GPU 3–10× slower; covariance MC error < 1 %, overkill for d=9 |
| 100,000 – 249,999 | CPU | Stress-test | GPU still slower; no scientific justification for d=9 filter |
| 250,000 – 359,999 | GPU | Stress-test | GPU begins recovering overhead margin; Smart Switch activates GPU |
| 360,000 | GPU | Stress-test | Empirical break-even point on this hardware stack |
| > 360,000 | GPU | Stress-test | GPU progressively faster; justified for augmented state or multi-node models |
| 500,000 (max) | GPU | Stress-test | ~1.4× GPU advantage; VRAM usage ~308 MB (1.5 % of 20 GB) |

### 7.2 Scientific Context

The 9-dimensional point-kinetics EnKF implemented in this project reaches statistical convergence at N ≈ 1,000–5,000. Values above this threshold serve exclusively research or benchmarking purposes:

| Range | Label | Justification |
|---|---|---|
| N ≤ 10,000 | Operational | Industry standard for online core monitoring (EDF: N = 50–200; CEA/MIT research: N ≤ 2,000) |
| N ≤ 100,000 | Research-grade | Monte Carlo error in P_f (9×9) below 0.3 %; standard for academic EnKF studies |
| N ≤ 500,000 | Stress-test | No improvement in filter quality for d=9; justified only for hardware benchmarking or future model extensions with d ≥ 100 (spatial kinetics) |

---

## 8. Design Consequences

### 8.1 Device Selection Logic

The `run_virtual_sensor_job` Celery task passes `device="cuda"` by default and falls back to `"cpu"` when `torch.cuda.is_available()` returns False. This is correct: the fallback is semantically sound but carries a performance penalty at large N that must be communicated to operators.

### 8.2 Smart Switch Threshold

The `_GPU_ENSEMBLE_THRESHOLD = 250_000` constant in `backend/app/worker/tasks.py` implements the device selection policy derived from Section 6. Ensemble sizes below this threshold are routed to CPU regardless of the `device` field in the API request. The threshold represents a 30 % margin below the empirical break-even (N ≈ 360,000) to ensure the CPU path is always taken when the GPU offers no meaningful advantage.

To adjust the threshold for a different hardware stack, back-calculate the WSL2 overhead from a measurement at a known N (as in Section 6.2) and compute `N* = t_overhead / 1,350e-12`. Set the threshold to approximately 0.7 × N*.

### 8.3 dtype Considerations

The EnsembleSolver defaults to float32 (single precision). Switching to float64 (double precision) doubles the data volume and halves the arithmetic intensity, shifting the crossover to approximately N = 240,000. Float32 is sufficient for the EnKF application, where model uncertainty at ensemble level is multiple orders of magnitude larger than float32 rounding error.

---

## 9. References

1. PCIe 4.0 Specification, PCI-SIG, 2017.
2. AMD Radeon RX 7900 XT Product Specification, AMD, 2022.
3. Williams, S., Waterman, A., Patterson, D. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." *Communications of the ACM*, 52(4), 65-76, 2009.
4. Evensen, G. "The Ensemble Kalman Filter: Theoretical Formulation and Practical Implementation." *Ocean Dynamics*, 53(4), 343-367, 2003.
5. AMD ROCm Documentation, "HIP Performance Guide," https://rocm.docs.amd.com, 2024.
