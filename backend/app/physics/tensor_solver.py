"""
Virtual Sensor — Ensemble solver for PWR Point-Kinetics + Thermal-Hydraulics.

Architecture
------------
Two public classes are provided:

EnsembleSolver  (primary — Data Assimilation)
    Initialises an ensemble of N members from a single nominal ReactorParams
    by adding independent Gaussian perturbations to selected parameters and
    initial states.  All N members are integrated simultaneously with a
    vectorised in-place RK4 step — no Python loops over N.

    Supports an Ensemble Kalman Filter (EnKF) assimilation step so that
    sensor measurements can correct the ensemble in real time.

TensorSolver  (legacy — parameter batch)
    Fixed-step RK4 for N *independent* reactor configurations (parameter
    sweeps, uncertainty quantification).  Preserves the ReactorSimulator ABC
    so the rest of the application is unaffected.

Both classes are strictly PyTorch.  The NumPy compatibility adapter has been
removed.  device='cuda' maps transparently to ROCm/HIP on AMD GPUs.

Physical model  (same equations as base.py)
-------------------------------------------
State vector  y ∈ ℝ⁹  per member:
    y[0]      n        normalised neutron population      [-]
    y[1..6]   C₁..C₆  delayed-neutron precursor groups   [-]
    y[7]      T_f      average fuel temperature            [K]
    y[8]      T_c      average coolant temperature         [K]

ODEs:
    dn/dt   = [(ρ − β) / Λ] · n  +  Σᵢ λᵢ Cᵢ
    dCᵢ/dt  = (βᵢ / Λ) · n  −  λᵢ Cᵢ
    ρ       = ρ_ext + α_f (T_f − T_f0) + α_c (T_c − T_c0)
    γ_f dT_f/dt = P − UA_fc (T_f − T_c),   P = n · P₀
    γ_c dT_c/dt = UA_fc (T_f − T_c) − G_cool (T_c − T_in)

Memory budget  (float32, N = 100 000, n_steps = 1 000)
------------------------------------------------------
  RK4 state + scratch   6 × (N,9)  × 4 B  ≈  22 MB
  RHS scratch tensors   6 × (N,)   × 4 B  ≈   2 MB
  Perturbed param vecs  6 × (N,)   × 4 B  ≈   2 MB
  Shared param views    β, λ stored as (1,6) expanded  ≈  <1 MB
  Stats history (CPU)   2 × n_steps × 9  × 8 B  ≈ negligible
  ─────────────────────────────────────────────────────────────
  Total                                          ≈  27 MB  ✓
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from app.physics.base import ReactorParams, ReactorSimulator, SimulationResult

# Lazy torch import — the module loads without PyTorch; classes raise RuntimeError
# at instantiation time if torch is absent, providing a clear diagnostic message.
try:
    import torch
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    """Raise a descriptive RuntimeError when PyTorch is not installed."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed.  Install the GPU variant for ROCm:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2\n"
            "or for CUDA:\n"
            "  pip install torch\n"
            "Then add 'torch' to pyproject.toml [project.optional-dependencies] gpu."
        )


def _no_grad(fn: Any) -> Any:
    """
    Conditional @torch.no_grad() decorator.

    Wraps the function in torch.no_grad() when PyTorch is available;
    returns the function unmodified otherwise so the module loads cleanly
    in environments without torch installed.
    """
    if _TORCH_AVAILABLE:
        return torch.no_grad()(fn)
    return fn

# ── Constants ─────────────────────────────────────────────────────────────────

_DT_MAX: float = 0.01      # [s] RK4 stability ceiling for PWR PKE stiffness
_DEFAULT_N: int = 100_000  # default ensemble size

# Observable name → state-vector column index
_OBS_INDEX: dict[str, int] = {
    "neutron_population":  0,
    "fuel_temperature":    7,
    "coolant_temperature": 8,
}


# ── Noise configuration ───────────────────────────────────────────────────────

@dataclass
class NoiseConfig:
    """
    Perturbation widths for ensemble initialisation.

    Relative noise:  σ = σ_rel × |nominal|
    Absolute noise:  σ = σ_abs             (for temperatures in K)

    Set any field to 0.0 to disable perturbation for that variable.
    """
    # Parameter perturbations (relative)
    doppler_coefficient_rel:     float = 0.02   # 2 % of α_f
    fuel_coolant_conductance_rel: float = 0.02  # 2 % of UA_fc
    nominal_power_rel:           float = 0.01   # 1 % of P₀

    # Initial-state perturbations
    neutron_population_rel:      float = 0.005  # 0.5 % of n₀ = 1
    fuel_temp_abs:               float = 2.0    # ±2 K
    coolant_temp_abs:            float = 1.0    # ±1 K


# ── Ensemble result ───────────────────────────────────────────────────────────

@dataclass
class EnsembleResult:
    """
    Time-series statistics (mean ± std) over all N ensemble members.

    Storing per-member histories would require O(N × n_steps × 9) memory
    (~3.6 GB for N=100 000, n_steps=1 000, float32) — instead only the
    ensemble mean and standard deviation are returned.
    """
    time: list[float]

    mean_neutron_population:    list[float]
    std_neutron_population:     list[float]

    mean_fuel_temperature_K:    list[float]
    std_fuel_temperature_K:     list[float]

    mean_coolant_temperature_K: list[float]
    std_coolant_temperature_K:  list[float]

    mean_power_W:               list[float]
    std_power_W:                list[float]

    mean_reactivity:            list[float]
    std_reactivity:             list[float]

    ensemble_size: int
    device: str


# ── EnsembleSolver ────────────────────────────────────────────────────────────

class EnsembleSolver:
    """
    Vectorised ensemble integrator for the PWR PKE + TH system.

    All computation runs on ``device`` (GPU recommended).  No Python loops
    over the N ensemble members exist anywhere in the integration path.

    Typical workflow — open-loop forward propagation
    ------------------------------------------------
    solver = EnsembleSolver(N=100_000, device='cuda')
    result = solver.run_forward(params, time_span=(0.0, 100.0), dt=0.001)

    Typical workflow — closed-loop data assimilation
    -------------------------------------------------
    solver = EnsembleSolver(N=100_000, device='cuda')
    solver.initialize(params, noise=NoiseConfig())

    for t, measurement in sensor_stream:
        solver.step(dt)
        solver.assimilate(
            observation_values={'coolant_temperature': measurement},
            observation_noise_std={'coolant_temperature': 0.5},
        )
        mean, std = solver.state_mean, solver.state_std
    """

    # ──────────────────────────────────────────────────────── construction ──

    def __init__(
        self,
        N: int = _DEFAULT_N,
        device: str = "cuda",
        dtype: Any = None,
    ) -> None:
        """
        Parameters
        ----------
        N:
            Ensemble size (number of parallel members).
        device:
            PyTorch device string.  'cuda' works for both NVIDIA CUDA and AMD
            ROCm.  Use 'cpu' for debugging without a GPU.
        dtype:
            PyTorch dtype.  Defaults to torch.float32 (best GPU throughput).
            Pass torch.float64 for higher precision.
        """
        _require_torch()
        if dtype is None:
            dtype = torch.float32
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA / ROCm device requested but torch.cuda.is_available() "
                "returned False.  Install PyTorch with ROCm support or use "
                "device='cpu'."
            )
        self.N      = N
        self.device = device
        self.dtype  = dtype

        # ── State tensor (N, 9) — allocated by initialize() ────────────────
        self._y: torch.Tensor | None = None

        # ── RK4 scratch buffers (N, 9) — allocated once by initialize() ────
        self._k1:    torch.Tensor | None = None
        self._k2:    torch.Tensor | None = None
        self._k3:    torch.Tensor | None = None
        self._k4:    torch.Tensor | None = None
        self._y_tmp: torch.Tensor | None = None

        # ── RHS intermediate buffers (N,) ───────────────────────────────────
        self._rho:           torch.Tensor | None = None
        self._lambda_C_sum:  torch.Tensor | None = None
        self._Q_fc:          torch.Tensor | None = None
        self._tmp_a:         torch.Tensor | None = None   # scratch (N,)
        self._tmp_b:         torch.Tensor | None = None   # scratch (N,)

        # ── Per-member parameter tensors ─────────────────────────────────────
        # Set by initialize().  Shapes: (N,) or (N,6) views.
        self._beta_mat:          torch.Tensor | None = None  # (N,6)
        self._lam_mat:           torch.Tensor | None = None  # (N,6)
        self._beta_over_Lambda:  torch.Tensor | None = None  # (N,6)
        self._beta_total:        torch.Tensor | None = None  # (N,)
        self._Lambda:            torch.Tensor | None = None  # (N,)
        self._P0:                torch.Tensor | None = None  # (N,) perturbed
        self._T_f0:              torch.Tensor | None = None  # (N,)
        self._T_c0:              torch.Tensor | None = None  # (N,)
        self._T_in:              torch.Tensor | None = None  # (N,)
        self._gamma_f:           torch.Tensor | None = None  # (N,)
        self._gamma_c:           torch.Tensor | None = None  # (N,)
        self._UA_fc:             torch.Tensor | None = None  # (N,) perturbed
        self._G_cool:            torch.Tensor | None = None  # (N,)
        self._alpha_f:           torch.Tensor | None = None  # (N,) perturbed
        self._alpha_c:           torch.Tensor | None = None  # (N,)
        self._rho_ext:           torch.Tensor | None = None  # (N,)

        self._initialized = False

    # ──────────────────────────────────────────────────── private helpers ──

    def _alloc(self, *shape: int) -> torch.Tensor:
        """Empty tensor of given shape on self.device / self.dtype."""
        return torch.empty(shape, dtype=self.dtype, device=self.device)

    def _full(self, value: float) -> torch.Tensor:
        """Constant (N,) tensor."""
        return torch.full((self.N,), value, dtype=self.dtype, device=self.device)

    def _perturb_rel(
        self,
        nominal: float,
        rel_std: float,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """Return (N,) tensor: nominal × (1 + rel_std × ε),  ε ~ N(0,1)."""
        if rel_std == 0.0:
            return self._full(nominal)
        t = torch.randn(self.N, dtype=self.dtype, device=self.device, generator=rng)
        t.mul_(rel_std * abs(nominal)).add_(nominal)
        return t

    def _perturb_abs(
        self,
        nominal: float,
        abs_std: float,
        rng: torch.Generator,
    ) -> torch.Tensor:
        """Return (N,) tensor: nominal + abs_std × ε,  ε ~ N(0,1)."""
        if abs_std == 0.0:
            return self._full(nominal)
        t = torch.randn(self.N, dtype=self.dtype, device=self.device, generator=rng)
        t.mul_(abs_std).add_(nominal)
        return t

    # ──────────────────────────────────────────────────────── initialize ──

    def initialize(
        self,
        params: ReactorParams,
        noise: NoiseConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        """
        Build the ensemble from a single nominal ReactorParams.

        All major tensors are allocated here and reused for every subsequent
        call to ``step()``.  Call ``initialize()`` again to reset the ensemble.

        Perturbed per-member variables
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Parameter space:
          • α_f   (Doppler coefficient)      — relative Gaussian noise
          • UA_fc (fuel-coolant conductance)  — relative Gaussian noise
          • P₀    (nominal thermal power)     — relative Gaussian noise

        State space (initial conditions):
          • n₀    (neutron population)        — relative Gaussian noise
          • T_f₀  (fuel temperature)          — absolute Gaussian noise [K]
          • T_c₀  (coolant temperature)       — absolute Gaussian noise [K]

        All other parameters are identical across members and stored as (1,6)
        or (1,) tensors expanded to (N,6)/(N,) with zero extra VRAM cost.

        Parameters
        ----------
        params:
            Nominal reactor configuration.
        noise:
            Noise levels; NoiseConfig() defaults are used if None.
        seed:
            Optional RNG seed for reproducible ensembles.
        """
        if noise is None:
            noise = NoiseConfig()

        N, dev, dtyp = self.N, self.device, self.dtype

        rng = torch.Generator(device=dev)
        if seed is not None:
            rng.manual_seed(seed)

        # ── β and λ groups — uniform across members (expand = zero-copy) ───
        beta_np = np.asarray(params.beta_groups,   dtype=np.float32)
        lam_np  = np.asarray(params.lambda_groups, dtype=np.float32)
        beta_row = torch.tensor(beta_np, dtype=dtyp, device=dev).unsqueeze(0)  # (1,6)
        lam_row  = torch.tensor(lam_np,  dtype=dtyp, device=dev).unsqueeze(0)  # (1,6)

        self._beta_mat         = beta_row.expand(N, 6)   # (N,6) — no copy
        self._lam_mat          = lam_row.expand(N, 6)    # (N,6) — no copy

        beta_total_nom = float(sum(params.beta_groups))
        Lambda_nom     = float(params.prompt_neutron_lifetime)

        self._beta_total       = self._full(beta_total_nom)
        self._Lambda           = self._full(Lambda_nom)
        self._beta_over_Lambda = (beta_row / Lambda_nom).expand(N, 6)  # (N,6) no copy

        # ── Perturbed parameters (each member gets its own value) ───────────
        self._alpha_f = self._perturb_rel(
            params.doppler_coefficient,      noise.doppler_coefficient_rel,     rng
        )
        self._UA_fc   = self._perturb_rel(
            params.fuel_coolant_conductance, noise.fuel_coolant_conductance_rel, rng
        )
        self._P0      = self._perturb_rel(
            params.nominal_power,            noise.nominal_power_rel,           rng
        )

        # ── Fixed parameters (constant across members) ──────────────────────
        self._alpha_c = self._full(params.moderator_coefficient)
        self._rho_ext = self._full(params.external_reactivity)
        self._gamma_f = self._full(params.fuel_heat_capacity)
        self._gamma_c = self._full(params.coolant_heat_capacity)
        self._G_cool  = self._full(params.coolant_flow_capacity)
        self._T_in    = self._full(params.coolant_inlet_temp)
        self._T_f0    = self._full(params.nominal_fuel_temp)
        self._T_c0    = self._full(params.nominal_coolant_temp)

        # ── Initial state y₀  (N, 9) ────────────────────────────────────────
        # Steady-state precursors: Cᵢ₀ = βᵢ / (λᵢ · Λ)  — same for all members
        C0_row = beta_row / (lam_row * Lambda_nom)   # (1,6)

        n0  = self._perturb_rel(1.0,                          noise.neutron_population_rel, rng)
        Tf0 = self._perturb_abs(params.nominal_fuel_temp,    noise.fuel_temp_abs,          rng)
        Tc0 = self._perturb_abs(params.nominal_coolant_temp, noise.coolant_temp_abs,        rng)

        # Allocate or reuse state buffer
        if self._y is None or self._y.shape != (N, 9):
            self._y = self._alloc(N, 9)

        self._y[:, 0]   = n0
        self._y[:, 1:7] = C0_row.expand(N, 6)   # (N,6) — all members same C0
        self._y[:, 7]   = Tf0
        self._y[:, 8]   = Tc0

        # ── RK4 scratch buffers — allocate once ─────────────────────────────
        for attr in ("_k1", "_k2", "_k3", "_k4", "_y_tmp"):
            existing: torch.Tensor | None = getattr(self, attr)
            if existing is None or existing.shape != (N, 9):
                setattr(self, attr, self._alloc(N, 9))

        # ── RHS intermediate buffers ─────────────────────────────────────────
        for attr in ("_rho", "_lambda_C_sum", "_Q_fc", "_tmp_a", "_tmp_b"):
            existing = getattr(self, attr)
            if existing is None or existing.shape != (N,):
                setattr(self, attr, self._alloc(N))

        self._initialized = True

    # ───────────────────────────────────────────── right-hand side (RHS) ──

    def _rhs(self, y: torch.Tensor, out: torch.Tensor) -> None:
        """
        Compute dy/dt = f(y) for all N members simultaneously, writing into
        the pre-allocated ``out`` tensor.  Zero allocations inside this method
        (all intermediates reuse pre-allocated scratch buffers).

        Parameters
        ----------
        y   : (N, 9) — current ensemble state  [read-only]
        out : (N, 9) — overwritten with dy/dt
        """
        # ── Unpack state views (no copy) ────────────────────────────────────
        n   = y[:, 0]    # (N,)
        C   = y[:, 1:7]  # (N,6) — view
        T_f = y[:, 7]    # (N,)
        T_c = y[:, 8]    # (N,)

        # ── ρ = ρ_ext + α_f·(T_f−T_f0) + α_c·(T_c−T_c0) ──────────────────
        # _rho  ←  α_f · (T_f − T_f0)
        torch.sub(T_f, self._T_f0, out=self._rho)
        self._rho.mul_(self._alpha_f)
        self._rho.add_(self._rho_ext)

        # _tmp_a ←  α_c · (T_c − T_c0)
        torch.sub(T_c, self._T_c0, out=self._tmp_a)
        self._tmp_a.mul_(self._alpha_c)
        self._rho.add_(self._tmp_a)   # _rho now holds full ρ

        # ── Σᵢ λᵢ Cᵢ  →  (N,) ───────────────────────────────────────────────
        # elementwise (N,6) × (N,6) then sum along groups → (N,)
        torch.sum(self._lam_mat * C, dim=1, out=self._lambda_C_sum)

        # ── dn/dt ────────────────────────────────────────────────────────────
        # = ((ρ − β) / Λ) · n  +  Σλ_i C_i
        # _tmp_a ← (ρ − β) / Λ
        torch.sub(self._rho, self._beta_total, out=self._tmp_a)
        self._tmp_a.div_(self._Lambda)
        self._tmp_a.mul_(n)
        self._tmp_a.add_(self._lambda_C_sum)
        out[:, 0] = self._tmp_a

        # ── dCᵢ/dt = (βᵢ/Λ)·n − λᵢ·Cᵢ  →  (N,6) ───────────────────────────
        # broadcast n as (N,1) against (N,6) beta_over_Lambda
        out[:, 1:7] = self._beta_over_Lambda * n.unsqueeze(1) - self._lam_mat * C

        # ── Thermal power  P = n · P₀  (N,) ─────────────────────────────────
        torch.mul(n, self._P0, out=self._tmp_b)      # _tmp_b = P

        # ── Q_fc = UA_fc · (T_f − T_c)  (N,) ────────────────────────────────
        torch.sub(T_f, T_c, out=self._Q_fc)
        self._Q_fc.mul_(self._UA_fc)

        # ── dT_f/dt = (P − Q_fc) / γ_f ───────────────────────────────────────
        torch.sub(self._tmp_b, self._Q_fc, out=out[:, 7])
        out[:, 7].div_(self._gamma_f)

        # ── dT_c/dt = (Q_fc − G_cool·(T_c − T_in)) / γ_c ────────────────────
        # _tmp_a ← G_cool · (T_c − T_in)
        torch.sub(T_c, self._T_in, out=self._tmp_a)
        self._tmp_a.mul_(self._G_cool)
        torch.sub(self._Q_fc, self._tmp_a, out=out[:, 8])
        out[:, 8].div_(self._gamma_c)

    # ─────────────────────────────────────────────────────────── RK4 step ──

    def step(self, dt: float) -> None:
        """
        Advance all N ensemble members by one classical RK4 step, in-place.

        All six (N,9) scratch tensors are reused; no heap allocation occurs
        during this method.  Suitable for tight integration loops.

        Parameters
        ----------
        dt : float
            Integration step size [s].  Must satisfy dt ≤ 0.01 s for RK4
            stability with PWR-typical stiffness (|s₀| ≈ 325 s⁻¹).
        """
        assert self._initialized, "Call initialize() before step()."

        y, k1, k2, k3, k4, y_tmp = (
            self._y, self._k1, self._k2, self._k3, self._k4, self._y_tmp
        )
        half  = dt * 0.5
        sixth = dt / 6.0

        # k1 = f(y)
        self._rhs(y, k1)

        # k2 = f(y + h/2 · k1)
        torch.add(y, k1, alpha=half, out=y_tmp)
        self._rhs(y_tmp, k2)

        # k3 = f(y + h/2 · k2)
        torch.add(y, k2, alpha=half, out=y_tmp)
        self._rhs(y_tmp, k3)

        # k4 = f(y + h · k3)
        torch.add(y, k3, alpha=dt, out=y_tmp)
        self._rhs(y_tmp, k4)

        # y ← y + h/6 · (k1 + 2k2 + 2k3 + k4)
        # Accumulate into y_tmp to avoid polluting k1..k4
        torch.add(k1, k4, out=y_tmp)            # k1 + k4
        y_tmp.add_(k2, alpha=2.0)               # + 2k2
        y_tmp.add_(k3, alpha=2.0)               # + 2k3
        y.add_(y_tmp, alpha=sixth)              # y += h/6 · (...)

    # ──────────────────────────────────────────── EnKF assimilation step ──

    @_no_grad
    def assimilate(
        self,
        observation_values: dict[str, float],
        observation_noise_std: dict[str, float],
    ) -> None:
        """
        Stochastic Ensemble Kalman Filter (EnKF) observation update.

        For each scalar observation, the full state vector of every ensemble
        member is updated using the standard stochastic EnKF equations:

            Kalman gain (per state component j):
                K_j = Cov(xⱼ, H·x) / (Var(H·x) + R)

            Per-member update:
                xᵢ ← xᵢ + K · (y_obs + εᵢ − H·xᵢ),   εᵢ ~ N(0, R)

        Observations are processed sequentially; for correlated sensors use
        a single call with all observations (they are still treated as
        independent here — extend to joint update for correlated sensors).

        Supported observable names
        --------------------------
        'neutron_population'  →  y[:, 0]
        'fuel_temperature'    →  y[:, 7]  [K]
        'coolant_temperature' →  y[:, 8]  [K]

        Parameters
        ----------
        observation_values:
            {observable_name: scalar measurement}
        observation_noise_std:
            {observable_name: 1-sigma measurement noise [same units as obs]}
        """
        assert self._initialized, "Call initialize() before assimilate()."

        for obs_name, y_obs_scalar in observation_values.items():
            if obs_name not in _OBS_INDEX:
                raise ValueError(
                    f"Unknown observable '{obs_name}'.  "
                    f"Supported: {list(_OBS_INDEX.keys())}"
                )
            idx = _OBS_INDEX[obs_name]
            R   = float(observation_noise_std[obs_name]) ** 2   # obs noise variance

            H_y   = self._y[:, idx]          # (N,) — ensemble values for this obs
            y_bar = H_y.mean()               # scalar
            A_obs = H_y - y_bar              # (N,) — obs-space anomalies

            # Ensemble variance of the observable
            var_Hy = (A_obs * A_obs).mean()  # scalar

            # Stochastic observation perturbation: εᵢ ~ N(0, R)
            obs_noise = torch.randn(
                self.N, dtype=self.dtype, device=self.device
            ).mul_(math.sqrt(R))

            # Innovation per member
            innovation = (y_obs_scalar + obs_noise) - H_y  # (N,)

            # Cross-covariance  PHᵀ = (1/N) Σ (xᵢ − x̄)(H·xᵢ − H·x̄) → (9,)
            x_bar  = self._y.mean(dim=0, keepdim=True)  # (1,9)
            A_full = self._y - x_bar                    # (N,9) state anomalies
            PHT    = (A_full * A_obs.unsqueeze(1)).mean(dim=0)  # (9,)

            # Full-state Kalman gain vector  K ∈ ℝ⁹
            denom  = var_Hy + R
            K_full = PHT / denom                        # (9,)

            # State update: xᵢ += K_j · innovation_i  for all j simultaneously
            # Outer product: (N,1) × (1,9) → (N,9) increment
            self._y.addcmul_(
                innovation.unsqueeze(1),   # (N,1)
                K_full.unsqueeze(0),       # (1,9)
            )

    # ──────────────────────────────────────────────────────── statistics ──

    @property
    def state_mean(self) -> torch.Tensor:
        """Current ensemble mean of the state vector, shape (9,), on device."""
        assert self._initialized
        return self._y.mean(dim=0)

    @property
    def state_std(self) -> torch.Tensor:
        """Current ensemble standard deviation, shape (9,), on device."""
        assert self._initialized
        return self._y.std(dim=0, correction=1)

    @_no_grad
    def current_reactivity(self) -> torch.Tensor:
        """
        Reactivity ρ for all N members, shape (N,), using each member's own α_f.

        Returns a tensor on self.device.
        """
        assert self._initialized
        T_f = self._y[:, 7]
        T_c = self._y[:, 8]
        return (
            self._rho_ext
            + self._alpha_f * (T_f - self._T_f0)
            + self._alpha_c * (T_c - self._T_c0)
        )

    # ──────────────────────────────────────────────────────── run_forward ──

    @_no_grad
    def run_forward(
        self,
        params: ReactorParams,
        time_span: tuple[float, float],
        dt: float,
        *,
        noise: NoiseConfig | None = None,
        seed: int | None = None,
    ) -> EnsembleResult:
        """
        Initialise the ensemble and integrate forward, returning time-series
        statistics (mean and std) at each output step.

        Per-member full histories are NOT stored to keep memory bounded.
        Running mean and std are computed via two passes over the (N,9) state
        at each step — each requiring only O(N) GPU work.

        Parameters
        ----------
        params:
            Nominal reactor configuration.
        time_span:
            (t_start, t_end) [s].
        dt:
            Fixed integration step [s].  Must satisfy dt ≤ 0.01.
        noise:
            Ensemble perturbation config.  Defaults to NoiseConfig().
        seed:
            RNG seed for reproducibility.

        Returns
        -------
        EnsembleResult
        """
        t0, tf = time_span
        _validate_time_args(t0, tf, dt)

        self.initialize(params, noise=noise, seed=seed)

        # ── Time grid ────────────────────────────────────────────────────────
        t_np    = np.arange(t0, tf + dt * 0.5, dt)
        t_np    = t_np[t_np <= tf]
        n_steps = len(t_np)

        # ── Pre-allocate statistics arrays (CPU, tiny) ───────────────────────
        mean_hist = np.empty((n_steps, 9), dtype=np.float64)
        std_hist  = np.empty((n_steps, 9), dtype=np.float64)

        # ── Record t₀ ────────────────────────────────────────────────────────
        _record_stats(self._y, mean_hist, std_hist, 0)

        # ── Time-integration loop (loops over steps, NOT over N) ─────────────
        for step_idx in range(1, n_steps):
            self.step(dt)
            _record_stats(self._y, mean_hist, std_hist, step_idx)

        # ── Derive power and reactivity statistics from state statistics ─────
        #   Power:     P = n · P₀  (linear in n; P₀ variation is a 2nd-order effect)
        #   Reactivity: ρ ≈ ρ_ext + α_f·ΔT_f + α_c·ΔT_c  (linear approximation)
        P0_nom   = float(params.nominal_power)
        alpha_f0 = float(params.doppler_coefficient)
        alpha_c0 = float(params.moderator_coefficient)
        rho_ext0 = float(params.external_reactivity)
        Tf0_nom  = float(params.nominal_fuel_temp)
        Tc0_nom  = float(params.nominal_coolant_temp)

        n_mean  = mean_hist[:, 0]
        n_std   = std_hist[:, 0]
        Tf_mean = mean_hist[:, 7]
        Tc_mean = mean_hist[:, 8]
        Tf_std  = std_hist[:, 7]
        Tc_std  = std_hist[:, 8]

        pwr_mean = n_mean * P0_nom
        pwr_std  = n_std  * P0_nom

        rho_mean = (
            rho_ext0
            + alpha_f0 * (Tf_mean - Tf0_nom)
            + alpha_c0 * (Tc_mean - Tc0_nom)
        )
        # Error propagation for linear ρ (assuming T_f, T_c uncorrelated)
        rho_std = np.sqrt(
            (alpha_f0 * Tf_std) ** 2 + (alpha_c0 * Tc_std) ** 2
        )

        return EnsembleResult(
            time=t_np.tolist(),
            mean_neutron_population=n_mean.tolist(),
            std_neutron_population=n_std.tolist(),
            mean_fuel_temperature_K=Tf_mean.tolist(),
            std_fuel_temperature_K=Tf_std.tolist(),
            mean_coolant_temperature_K=Tc_mean.tolist(),
            std_coolant_temperature_K=Tc_std.tolist(),
            mean_power_W=pwr_mean.tolist(),
            std_power_W=pwr_std.tolist(),
            mean_reactivity=rho_mean.tolist(),
            std_reactivity=rho_std.tolist(),
            ensemble_size=self.N,
            device=self.device,
        )


# ── TensorSolver ─────────────────────────────────────────────────────────────

class TensorSolver(ReactorSimulator):
    """
    Fixed-step vectorised RK4 for N **independent** reactor configurations.

    Preserves the ``ReactorSimulator`` ABC (``run_simulation`` / ``run_batch``)
    so existing application code requires no changes.

    This class is strictly PyTorch — the NumPy compatibility adapter from the
    previous implementation has been removed.  CPU is still supported via
    device='cpu'.

    Parameters
    ----------
    device:
        PyTorch device.  Auto-detects 'cuda' when available, falls back to
        'cpu'.  Pass explicitly to override.
    """

    def __init__(self, device: str | None = None) -> None:
        _require_torch()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype  = torch.float64   # float64 for per-run accuracy

    @property
    def backend(self) -> str:
        return f"torch[{self.device}]"

    # ── ReactorSimulator ABC ──────────────────────────────────────────────────

    def run_simulation(
        self,
        params: ReactorParams,
        time_span: tuple[float, float],
        dt: float,
    ) -> SimulationResult:
        """Single-run convenience wrapper — delegates to run_batch(N=1)."""
        return self.run_batch([params], time_span, dt)[0]

    def run_batch(
        self,
        params_list: Sequence[ReactorParams],
        time_span: tuple[float, float],
        dt: float,
    ) -> list[SimulationResult]:
        """
        Simulate N independent reactor configurations simultaneously.

        All N configurations share ``time_span`` and ``dt`` but may have
        completely different physical parameters.

        Parameters
        ----------
        params_list:
            Sequence of ReactorParams.  Length N ≥ 1.
        time_span:
            (t_start, t_end) [s].
        dt:
            Fixed integration step [s].  Must satisfy dt ≤ 0.01.

        Returns
        -------
        list[SimulationResult]
            One result per input parameter set, in order.
        """
        t0, tf = time_span
        _validate_time_args(t0, tf, dt)
        if not params_list:
            return []

        dev, dtyp = self.device, self.dtype
        N = len(params_list)

        # ── Build per-member parameter tensors ──────────────────────────────
        def _col(vals: list[float]) -> torch.Tensor:
            return torch.tensor(vals, dtype=dtyp, device=dev)

        beta_mat = torch.tensor(
            [p.beta_groups    for p in params_list], dtype=dtyp, device=dev
        )  # (N,6)
        lam_mat  = torch.tensor(
            [p.lambda_groups  for p in params_list], dtype=dtyp, device=dev
        )  # (N,6)

        beta_total = _col([sum(p.beta_groups)              for p in params_list])
        Lambda     = _col([p.prompt_neutron_lifetime       for p in params_list])
        P0         = _col([p.nominal_power                 for p in params_list])
        T_f0       = _col([p.nominal_fuel_temp             for p in params_list])
        T_c0       = _col([p.nominal_coolant_temp          for p in params_list])
        T_in       = _col([p.coolant_inlet_temp            for p in params_list])
        gamma_f    = _col([p.fuel_heat_capacity            for p in params_list])
        gamma_c    = _col([p.coolant_heat_capacity         for p in params_list])
        UA_fc      = _col([p.fuel_coolant_conductance      for p in params_list])
        G_cool     = _col([p.coolant_flow_capacity         for p in params_list])
        alpha_f    = _col([p.doppler_coefficient           for p in params_list])
        alpha_c    = _col([p.moderator_coefficient         for p in params_list])
        rho_ext    = _col([p.external_reactivity           for p in params_list])

        beta_over_Lambda = beta_mat / Lambda.unsqueeze(1)  # (N,6)

        # ── Initial state y₀ (N, 9) ─────────────────────────────────────────
        beta_np = np.array([p.beta_groups          for p in params_list], dtype=np.float64)
        lam_np  = np.array([p.lambda_groups        for p in params_list], dtype=np.float64)
        L_np    = np.array([p.prompt_neutron_lifetime for p in params_list], dtype=np.float64)
        C0_np   = beta_np / (lam_np * L_np[:, None])  # (N,6)

        y0_np          = np.empty((N, 9), dtype=np.float64)
        y0_np[:, 0]    = 1.0
        y0_np[:, 1:7]  = C0_np
        y0_np[:, 7]    = [p.nominal_fuel_temp    for p in params_list]
        y0_np[:, 8]    = [p.nominal_coolant_temp for p in params_list]

        y = torch.tensor(y0_np, dtype=dtyp, device=dev)  # (N,9)

        # ── RHS closure (captures parameter tensors; no loops over N) ───────
        def rhs(state: torch.Tensor) -> torch.Tensor:
            n_s   = state[:, 0]
            C     = state[:, 1:7]
            T_f   = state[:, 7]
            T_c   = state[:, 8]
            rho   = rho_ext + alpha_f * (T_f - T_f0) + alpha_c * (T_c - T_c0)
            lam_C = (lam_mat * C).sum(dim=1)
            dn    = ((rho - beta_total) / Lambda) * n_s + lam_C
            dC    = beta_over_Lambda * n_s.unsqueeze(1) - lam_mat * C
            Q_fc  = UA_fc * (T_f - T_c)
            dTf   = (n_s * P0 - Q_fc) / gamma_f
            dTc   = (Q_fc - G_cool * (T_c - T_in)) / gamma_c
            return torch.stack(
                [dn,
                 dC[:, 0], dC[:, 1], dC[:, 2],
                 dC[:, 3], dC[:, 4], dC[:, 5],
                 dTf, dTc],
                dim=1,
            )

        # ── Time grid ────────────────────────────────────────────────────────
        t_np    = np.arange(t0, tf + dt * 0.5, dt)
        t_np    = t_np[t_np <= tf]
        n_steps = len(t_np)

        # Pre-allocate history on CPU (n_steps, N, 9) — float64
        history = np.empty((n_steps, N, 9), dtype=np.float64)
        history[0] = y.cpu().numpy()

        # ── Integration (time loop; N dimension fully vectorised) ────────────
        for step_idx in range(1, n_steps):
            y = _rk4_standalone(rhs, y, dt)
            history[step_idx] = y.cpu().numpy()

        # ── Pack SimulationResult per member ─────────────────────────────────
        results: list[SimulationResult] = []
        for i, p in enumerate(params_list):
            traj  = history[:, i, :]   # (n_steps, 9)
            n_arr = traj[:, 0]
            C_mat = traj[:, 1:7]
            Tf    = traj[:, 7]
            Tc    = traj[:, 8]
            rho_out = (
                p.external_reactivity
                + p.doppler_coefficient   * (Tf - p.nominal_fuel_temp)
                + p.moderator_coefficient * (Tc - p.nominal_coolant_temp)
            )
            results.append(SimulationResult(
                time=t_np.tolist(),
                neutron_population=n_arr.tolist(),
                power_W=(n_arr * p.nominal_power).tolist(),
                fuel_temperature_K=Tf.tolist(),
                coolant_temperature_K=Tc.tolist(),
                reactivity=rho_out.tolist(),
                precursor_concentrations=C_mat.tolist(),
            ))

        return results


# ── Module-level helpers ─────────────────────────────────────────────────────

def _rk4_standalone(
    rhs,  # Callable[[Tensor], Tensor]
    y: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Classical RK4 returning a new tensor.  Used by TensorSolver."""
    k1 = rhs(y)
    k2 = rhs(y + (dt * 0.5) * k1)
    k3 = rhs(y + (dt * 0.5) * k2)
    k4 = rhs(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _validate_time_args(t0: float, tf: float, dt: float) -> None:
    if tf <= t0:
        raise ValueError(f"time_span: t_end={tf} must be > t_start={t0}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if dt > _DT_MAX:
        raise ValueError(
            f"dt={dt:.4g} s exceeds the RK4 stability limit ({_DT_MAX} s) "
            "for the stiff PWR PKE system.  Use dt ≤ 0.01 s or switch to "
            "ScipySolver for adaptive step-size control."
        )


def _record_stats(
    y: torch.Tensor,
    mean_arr: np.ndarray,
    std_arr: np.ndarray,
    step: int,
) -> None:
    """
    Write mean and std of the (N, 9) state into pre-allocated (n_steps, 9)
    NumPy arrays at row ``step``.

    One GPU→CPU transfer of 18 float32/float64 scalars per time step.
    Bessel-corrected std (correction=1) matches np.std(ddof=1).
    """
    mean_arr[step] = y.mean(dim=0).cpu().numpy()
    std_arr[step]  = y.std(dim=0, correction=1).cpu().numpy()
