"""
Ensemble Kalman Filter (EnKF) — Data Assimilation motor for the PWR Virtual Sensor.

Theory
------
The EnKF approximates the optimal Bayesian filter by representing the probability
distribution of the reactor state as an ensemble of N Monte Carlo members.

State vector (matches EnsembleSolver convention):

    x ∈ ℝ⁹  =  [n,  C₁, C₂, C₃, C₄, C₅, C₆,  T_f,  T_c]
                 0   1   2   3   4   5   6    7     8

Physical insight
~~~~~~~~~~~~~~~~
In a real plant, T_coolant (col 8) is readily measurable (RTD sensors),
while T_fuel (col 7) is inaccessible.  The thermal coupling

    γ_f dT_f/dt = P − UA_fc (T_f − T_c)

creates a non-zero cross-covariance  Cov(T_f, T_c) ≠ 0  in the ensemble.
The EnKF exploits this correlation: when the measured T_coolant deviates
from the ensemble forecast, the Kalman gain propagates the correction into
the HIDDEN T_fuel state, effectively "seeing through" the fuel cladding.

EnKF cycle (one time step)
--------------------------
Let  X^k ∈ ℝ^{N×9}  denote the ensemble at assimilation step k,
     M(·)  denote the physical model (RK4 integrator),
     H ∈ ℝ^{m×9}  the observation operator (here m=1, selects T_coolant),
     y ∈ ℝ^m  the scalar measurement,
     R ∈ ℝ^{m×m}  the measurement noise covariance (here scalar).

  ┌─────────────────────────────────────────────────────────────────────┐
  │ (a)  FORECAST                                                       │
  │      x_f^i = M(x_a^{k−1,i},  dt),   i = 1 … N                    │
  │                                                                     │
  │ (b)  ENSEMBLE COVARIANCE                                           │
  │      x̄_f   = (1/N) Σᵢ x_f^i              ∈ ℝ⁹                   │
  │      A_f   = X_f − 1 x̄_f^T               ∈ ℝ^{N×9} (anomalies)  │
  │      P_f   = (1/(N−1)) A_f^T A_f          ∈ ℝ^{9×9}              │
  │                                                                     │
  │ (c)  INNOVATION COVARIANCE                                         │
  │      S = H P_f H^T + R                    ∈ ℝ^{m×m}              │
  │                                                                     │
  │ (d)  KALMAN GAIN  (solved via linalg, not direct inversion)        │
  │      K = P_f H^T S^{−1}                   ∈ ℝ^{9×m}              │
  │        ⟺  S K^T = H P_f  (transpose system)                       │
  │                                                                     │
  │ (e)  STOCHASTIC ANALYSIS UPDATE  (Hunt et al. 2007)                │
  │      ỹ^i  = y + ε^i,  ε^i ~ N(0, R)      ∈ ℝ^m  (perturbed obs) │
  │      d^i  = ỹ^i − H x_f^i                ∈ ℝ^m  (innovation)     │
  │      x_a^i = x_f^i + K d^i               ∈ ℝ⁹   (analysis)      │
  └─────────────────────────────────────────────────────────────────────┘

The perturbation of observations in step (e) is essential: it maintains the
rank of the analysis error covariance  P_a = (I − K H) P_f  and prevents
ensemble collapse (filter divergence) due to repeated assimilation.

Memory / performance budget
---------------------------
All EnKF tensors beyond the solver's own buffers are small:

    H          (1, 9)  × 4 B  =    36 B
    R          (1, 1)  × 4 B  =     4 B
    P_f        (9, 9)  × 4 B  =   324 B
    S          (1, 1)  × 4 B  =     4 B
    K          (9, 1)  × 4 B  =    36 B
    _eps       (N,)   × 4 B  =  0.4 MB   (pre-allocated scratch)
    _d         (N,)   × 4 B  =  0.4 MB   (pre-allocated scratch)
    ────────────────────────────────────
    Overhead beyond EnsembleSolver      ≈  0.8 MB   ✓
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ── PyTorch: same lazy-import pattern as tensor_solver.py ────────────────────
try:
    import torch
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required by EnKFSensor.  Install for ROCm:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2\n"
            "or for CUDA:  pip install torch"
        )


def _no_grad(fn: Any) -> Any:
    """Conditional @torch.no_grad() that works even when torch is absent."""
    if _TORCH_AVAILABLE:
        return torch.no_grad()(fn)
    return fn


# ── Local import (EnsembleSolver lives in the same package) ──────────────────
from app.physics.tensor_solver import EnsembleSolver  # noqa: E402

# ── State-vector column indices ───────────────────────────────────────────────
_IDX_N   = 0           # normalised neutron population  n
_IDX_C   = slice(1, 7) # delayed-neutron precursor groups C₁ … C₆
_IDX_T_F = 7           # average fuel temperature  T_f  [K]  ← HIDDEN
_IDX_T_C = 8           # average coolant temperature  T_c  [K]  ← OBSERVABLE


# ── Configuration dataclass ───────────────────────────────────────────────────

@dataclass
class EnKFConfig:
    """
    Tuning parameters for the Ensemble Kalman Filter.

    Attributes
    ----------
    obs_noise_var_K2:
        Measurement noise variance  R  [K²] for the T_coolant sensor.
        Typical RTD accuracy σ ≈ 0.3 – 0.5 K  →  R = 0.09 – 0.25 K².
        Default: 0.25 K²  (σ = 0.5 K).
    inflation_factor:
        Multiplicative covariance inflation applied to the ensemble anomalies
        before each assimilation step:

            A_f ← √α · A_f,   α ≥ 1

        Inflation counteracts ensemble spread collapse caused by repeated
        assimilation and model error.  α = 1.0 disables inflation.
        Typical range: 1.01 – 1.10.
    """
    obs_noise_var_K2: float = 0.25
    inflation_factor: float = 1.0


# ── Per-step diagnostic output ────────────────────────────────────────────────

@dataclass
class AssimilationStep:
    """
    Complete diagnostic record for one EnKF predict-update cycle.

    All temperatures in Kelvin; variance in K².
    """
    sim_time_s:     float   # simulation time after this step [s]
    dt_s:           float   # step size used [s]

    # ── Prior (forecast) ─────────────────────────────────────────────────────
    prior_T_fuel_mean_K:  float   # x̄_f[7]   — ensemble mean T_f before update
    prior_T_fuel_std_K:   float   # std(X_f[:, 7])

    prior_T_cool_mean_K:  float   # x̄_f[8]   — ensemble mean T_c before update
    prior_T_cool_std_K:   float   # std(X_f[:, 8])

    # ── Cross-covariance ─────────────────────────────────────────────────────
    cov_Tf_Tc_K2: float   # P_f[7, 8] — cross-covariance T_fuel ↔ T_coolant [K²]
    kalman_gain_T_fuel:  float   # K[7, 0]  — gain for the fuel temperature row

    # ── Observation ──────────────────────────────────────────────────────────
    y_obs_K:      float   # scalar measurement fed to the filter [K]
    innovation_mean_K: float  # mean(ỹ − H x_f) [K]

    # ── Posterior (analysis) ─────────────────────────────────────────────────
    posterior_T_fuel_mean_K: float  # x̄_a[7]  — ESTIMATED hidden fuel temperature
    posterior_T_fuel_var_K2: float  # Var(X_a[:, 7]) — uncertainty of the estimate


# ── Main filter class ─────────────────────────────────────────────────────────

class EnKFSensor:
    """
    Ensemble Kalman Filter that fuses T_coolant measurements with the
    physics ensemble to infer the hidden T_fuel state.

    Wraps an ``EnsembleSolver`` — the solver must already be initialised
    via ``solver.initialize(params, noise)`` before the first step.

    Example
    -------
    from app.physics.tensor_solver import EnsembleSolver, NoiseConfig
    from app.physics.assimilation  import EnKFSensor, EnKFConfig
    from app.physics.base          import ReactorParams

    params = ReactorParams(external_reactivity=50e-5)   # +50 pcm step
    solver = EnsembleSolver(N=100_000, device='cuda')
    solver.initialize(params, noise=NoiseConfig(), seed=0)

    sensor = EnKFSensor(solver, config=EnKFConfig(obs_noise_var_K2=0.25))

    # Real-time assimilation loop
    for t_meas, T_c_measured in plant_data_stream:
        T_f_est, T_f_var = sensor.step_assimilation(T_c_measured, dt=0.001)
        print(f"t={t_meas:.3f}s  T_fuel={T_f_est:.2f} ± {math.sqrt(T_f_var):.2f} K")
    """

    # ──────────────────────────────────────────────────────── construction ──

    def __init__(
        self,
        solver: EnsembleSolver,
        config: EnKFConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        solver:
            An ``EnsembleSolver`` instance whose ``initialize()`` method has
            already been called.  ``EnKFSensor`` directly mutates
            ``solver._y`` in-place at each assimilation step.
        config:
            Filter tuning.  Defaults to ``EnKFConfig()`` (σ_obs = 0.5 K,
            no covariance inflation).
        """
        _require_torch()

        if not solver._initialized:
            raise RuntimeError(
                "solver.initialize(params, noise) must be called before "
                "constructing EnKFSensor."
            )

        self.solver = solver
        self.config = config or EnKFConfig()

        dev  = solver.device
        dtyp = solver.dtype
        N    = solver.N

        # ── Observation matrix  H ∈ ℝ^{1×9}  — selects T_coolant ───────────
        # H @ x  extracts the observable: T_coolant = x[8].
        # H is a row vector with a single 1 at position _IDX_T_C.
        H = torch.zeros(1, 9, dtype=dtyp, device=dev)
        H[0, _IDX_T_C] = 1.0
        self._H: torch.Tensor = H  # (1, 9)

        # ── Measurement noise covariance  R ∈ ℝ^{1×1} ───────────────────────
        # R = σ_obs²  where σ_obs is the RTD sensor noise standard deviation.
        R_val = self.config.obs_noise_var_K2
        self._R: torch.Tensor = torch.tensor([[R_val]], dtype=dtyp, device=dev)
        self._sqrt_R: float   = math.sqrt(R_val)

        # ── Pre-allocated scratch tensors  (N,) ─────────────────────────────
        # Reused every step to avoid repeated heap allocation in the hot path.
        self._eps:     torch.Tensor = torch.empty(N, dtype=dtyp, device=dev)
        self._d:       torch.Tensor = torch.empty(N, dtype=dtyp, device=dev)

        # ── Simulation clock ─────────────────────────────────────────────────
        self._t: float = 0.0

        # ── Diagnostic history ───────────────────────────────────────────────
        self.history: list[AssimilationStep] = []

    # ───────────────────────────────────────────────── main EnKF cycle ──

    @_no_grad
    def step_assimilation(
        self,
        noisy_observation_T_coolant: float,
        dt: float,
    ) -> tuple[float, float]:
        """
        Execute one full EnKF predict-update cycle.

        Parameters
        ----------
        noisy_observation_T_coolant:
            Scalar RTD measurement of the coolant outlet temperature [K].
            The value may already include sensor noise; the filter accounts
            for it via the observation noise covariance R.
        dt:
            Integration time step [s].  Must satisfy dt ≤ 0.01 s for RK4
            stability with PWR PKE stiffness.

        Returns
        -------
        (T_fuel_mean, T_fuel_variance)
            Posterior estimate of the hidden average fuel temperature [K]
            and its posterior variance [K²] across the updated ensemble.
        """
        y_obs     = float(noisy_observation_T_coolant)
        H         = self._H          # (1, 9)
        R         = self._R          # (1, 1)
        N         = self.solver.N

        # ════════════════════════════════════════════════════════════════════
        # (a)  FORECAST STEP
        # ════════════════════════════════════════════════════════════════════
        #
        # Propagate each ensemble member forward by dt under the physics model:
        #
        #   x_f^i  =  M( x_a^{k−1,i},  dt ),    i = 1 … N
        #
        # M(·) is the classical 4th-order Runge-Kutta integrator implemented
        # in EnsembleSolver.step().  This is a fully vectorised operation over
        # all N members — no Python loop over N.
        # ════════════════════════════════════════════════════════════════════

        self.solver.step(dt)

        # Forecast ensemble matrix — direct view into solver state (no copy).
        # Shape: (N, 9)
        X_f: torch.Tensor = self.solver._y

        # ── Prior diagnostics (before update) ────────────────────────────────
        prior_Tf_mean = X_f[:, _IDX_T_F].mean().item()
        prior_Tf_std  = X_f[:, _IDX_T_F].std(correction=1).item()
        prior_Tc_mean = X_f[:, _IDX_T_C].mean().item()
        prior_Tc_std  = X_f[:, _IDX_T_C].std(correction=1).item()

        # ════════════════════════════════════════════════════════════════════
        # (b)  ENSEMBLE COVARIANCE  P_f ∈ ℝ^{9×9}
        # ════════════════════════════════════════════════════════════════════
        #
        # The sample (Bessel-corrected) covariance of the forecast ensemble:
        #
        #   P_f = (1/(N−1)) · A_f^T A_f
        #
        # where  A_f = X_f − 1·x̄_f^T  is the (N×9) anomaly matrix.
        #
        # torch.cov expects input shape (C, N) — transpose from (N, 9) to (9, N).
        # Returns (9, 9).
        # ════════════════════════════════════════════════════════════════════

        # Optional covariance inflation:  A_f ← √α · A_f
        # Applied directly to X_f by re-centering, scaling, and re-centering.
        if self.config.inflation_factor != 1.0:
            _inflate_ensemble(X_f, self.config.inflation_factor)

        #   P_f  ∈ ℝ^{9×9}
        P_f: torch.Tensor = torch.cov(X_f.T)   # (9, 9)

        # ── KEY PHYSICAL COUPLING ─────────────────────────────────────────────
        # The cross-covariance Cov(T_fuel, T_coolant) = P_f[7, 8].
        # This scalar is non-zero because UA_fc (fuel-coolant conductance)
        # thermally links the two nodes.  Its sign (positive for a PWR) tells
        # the filter: "when T_c is higher than forecast, T_f is likely higher too."
        cov_Tf_Tc: torch.Tensor = P_f[_IDX_T_F, _IDX_T_C]   # scalar tensor

        # ════════════════════════════════════════════════════════════════════
        # (c)  INNOVATION COVARIANCE  S ∈ ℝ^{1×1}
        # ════════════════════════════════════════════════════════════════════
        #
        # S combines the model's forecast uncertainty projected into observation
        # space with the measurement noise:
        #
        #   S = H P_f H^T + R     ∈ ℝ^{m×m}   (here m=1 → scalar wrapped in (1,1))
        #
        # H P_f H^T = Var_ensemble(H x_f^i) = forecast variance of T_coolant.
        # ════════════════════════════════════════════════════════════════════

        HPH_T: torch.Tensor = H @ P_f @ H.T    # (1, 9) @ (9, 9) @ (9, 1) → (1, 1)
        S:     torch.Tensor = HPH_T + R         # (1, 1)

        # ════════════════════════════════════════════════════════════════════
        # (d)  KALMAN GAIN  K ∈ ℝ^{9×1}
        # ════════════════════════════════════════════════════════════════════
        #
        # Optimal gain minimising the posterior error covariance:
        #
        #   K = P_f H^T S^{−1}
        #
        # Rather than computing S^{−1} explicitly (numerically unstable),
        # we solve the equivalent linear system:
        #
        #   S K^T = H P_f       (transpose both sides of  K S^T = P_f H^T)
        #
        # Since P_f and S are symmetric:  S^T = S,  P_f^T = P_f
        #
        #   S K^T = H P_f
        #
        # torch.linalg.solve(A, B) solves  A @ X = B.
        # Here:
        #   A = S           (1, 1)
        #   B = H @ P_f     (1, 9)   ← the right-hand side
        #   X = K^T         (1, 9)   ← solution
        # ════════════════════════════════════════════════════════════════════

        #   (1, 9) = H @ P_f  — projects P_f into observation space
        HP_f: torch.Tensor = H @ P_f                       # (1, 9)

        #   K^T ∈ ℝ^{1×9}  — Kalman gain transposed
        #   Solves the (1,1) system:  S @ K^T = H P_f
        K_T: torch.Tensor = torch.linalg.solve(S, HP_f)    # (1, 9)
        K:   torch.Tensor = K_T.T                          # (9, 1)

        # K[7, 0] is the gain for T_fuel:
        #   large  → measurement strongly corrects the fuel temperature estimate
        #   small  → fuel temp is uncertain but weakly coupled to the measurement
        gain_T_fuel: float = K[_IDX_T_F, 0].item()

        # ════════════════════════════════════════════════════════════════════
        # (e)  STOCHASTIC ANALYSIS UPDATE
        # ════════════════════════════════════════════════════════════════════
        #
        # Stochastic EnKF (Burgers et al. 1998): each member receives a
        # *different* random perturbation of the observation:
        #
        #   ỹ^i  = y_obs + ε^i,    ε^i ~ N(0, R),    i = 1 … N
        #
        # This is the crucial step that maintains the statistical consistency
        # of the posterior covariance  P_a ≈ (I − K H) P_f.
        # ════════════════════════════════════════════════════════════════════

        # Draw N independent observation noise samples in-place (no allocation).
        # self._eps ~ N(0, 1),  scaled by sqrt(R) to obtain N(0, R).
        torch.randn_(self._eps)                         # (N,) ~ N(0, 1)
        self._eps.mul_(self._sqrt_R)                    # (N,) ~ N(0, R)

        # Perturbed observations:  ỹ^i = y_obs + ε^i   ∈ ℝ^N
        # Written into pre-allocated _d buffer temporarily.
        torch.add(self._eps, y_obs, out=self._d)        # (N,) = y_obs + eps

        # Ensemble predicted observations:  ŷ^i = H x_f^i = T_c^i
        # Computed as a view — no copy, no allocation.
        H_X: torch.Tensor = X_f[:, _IDX_T_C]           # (N,) — view of X_f col 8

        # Innovation per member:  d^i = ỹ^i − ŷ^i
        # In-place:  _d ← _d − H_X
        self._d.sub_(H_X)                               # (N,) innovation

        innovation_mean: float = self._d.mean().item()

        # ── Analysis update: x_a^i = x_f^i + K d^i ─────────────────────────
        #
        # Matrix form for all N members at once:
        #
        #   X_a = X_f + d·K^T     ∈ ℝ^{N×9}
        #
        # where d·K^T is the outer product:  (N,1) @ (1,9) = (N,9)
        #
        # addmm_(mat1, mat2) computes:  self += mat1 @ mat2
        #   self  = X_f              (N, 9)
        #   mat1  = d.unsqueeze(1)   (N, 1)   ← innovation column vector
        #   mat2  = K_T              (1, 9)   ← Kalman gain row vector
        #
        # The outer product (N,1)@(1,9) simultaneously applies the gain to
        # every state variable of every member — no loops, no temporaries.
        X_f.addmm_(self._d.unsqueeze(1), K_T)          # in-place: (N,9) += (N,1)@(1,9)
        # Note: X_f IS solver._y (shared view) → solver state is now updated.

        # ════════════════════════════════════════════════════════════════════
        # OUTPUT — posterior T_fuel statistics
        # ════════════════════════════════════════════════════════════════════
        #
        # The posterior ensemble X_a[:, 7] is the filter's estimate of the
        # hidden fuel temperature distribution.
        #
        #   T̂_f  = (1/N) Σᵢ X_a[i, 7]          — point estimate
        #   σ²_f = (1/(N−1)) Σᵢ (X_a[i,7] − T̂_f)² — posterior variance
        # ════════════════════════════════════════════════════════════════════

        T_f_posterior: torch.Tensor = X_f[:, _IDX_T_F]     # (N,) — view

        T_f_mean: float = T_f_posterior.mean().item()
        T_f_var:  float = T_f_posterior.var(correction=1).item()

        # ── Record diagnostics ───────────────────────────────────────────────
        self._t += dt
        self.history.append(AssimilationStep(
            sim_time_s=self._t,
            dt_s=dt,
            prior_T_fuel_mean_K=prior_Tf_mean,
            prior_T_fuel_std_K=prior_Tf_std,
            prior_T_cool_mean_K=prior_Tc_mean,
            prior_T_cool_std_K=prior_Tc_std,
            cov_Tf_Tc_K2=cov_Tf_Tc.item(),
            kalman_gain_T_fuel=gain_T_fuel,
            y_obs_K=y_obs,
            innovation_mean_K=innovation_mean,
            posterior_T_fuel_mean_K=T_f_mean,
            posterior_T_fuel_var_K2=T_f_var,
        ))

        return T_f_mean, T_f_var

    # ──────────────────────────────────────────────── convenience helpers ──

    @_no_grad
    def full_state_estimate(self) -> dict[str, tuple[float, float]]:
        """
        Return the posterior mean and standard deviation for every state variable.

        Returns
        -------
        dict mapping variable name → (mean, std).

        Example
        -------
        {
          'neutron_population':  (1.00023, 0.00041),
          'precursor_C1':        (0.17382, 0.00012),
          ...
          'fuel_temperature_K':  (897.34,  1.83),
          'coolant_temperature_K': (595.21, 0.72),
        }
        """
        X = self.solver._y          # (N, 9)
        mu  = X.mean(dim=0)         # (9,)
        sig = X.std(dim=0, correction=1)  # (9,)

        mu_cpu  = mu.cpu().tolist()
        sig_cpu = sig.cpu().tolist()

        names = (
            ["neutron_population"]
            + [f"precursor_C{i}" for i in range(1, 7)]
            + ["fuel_temperature_K", "coolant_temperature_K"]
        )
        return {name: (m, s) for name, m, s in zip(names, mu_cpu, sig_cpu)}

    @_no_grad
    def ensemble_cross_covariance_matrix(self) -> torch.Tensor:
        """
        Return the current full 9×9 ensemble covariance matrix P.

        The element P[7, 8] is the cross-covariance Cov(T_fuel, T_coolant).
        The element P[7, 7] is Var(T_fuel) — the posterior fuel temperature
        uncertainty (useful for monitoring filter convergence).

        Returns a (9, 9) tensor on self.solver.device.
        """
        return torch.cov(self.solver._y.T)

    @_no_grad
    def run_assimilation(
        self,
        observations: list[float],
        dt: float,
    ) -> list[tuple[float, float]]:
        """
        Run the full assimilation loop over a sequence of measurements.

        Parameters
        ----------
        observations:
            List of T_coolant measurements [K], one per time step.
        dt:
            Fixed time step [s] between consecutive measurements.

        Returns
        -------
        list of (T_fuel_mean, T_fuel_variance) for each assimilation step.
        """
        results: list[tuple[float, float]] = []
        for y_obs in observations:
            results.append(self.step_assimilation(y_obs, dt))
        return results

    def reset_clock(self) -> None:
        """Reset the internal simulation clock to t=0 (does not reset the ensemble)."""
        self._t = 0.0

    def clear_history(self) -> None:
        """Discard the diagnostic history to free memory."""
        self.history = []

    @property
    def sim_time(self) -> float:
        """Elapsed simulation time [s]."""
        return self._t

    @property
    def T_fuel_estimate(self) -> tuple[float, float]:
        """
        Current posterior (mean, std) for T_fuel WITHOUT advancing time.

        Reads the current ensemble state — useful after the last assimilation
        step to query the estimate without doing another physics step.
        """
        T_f = self.solver._y[:, _IDX_T_F]
        return T_f.mean().item(), T_f.std(correction=1).item()


# ── Module-level utility ──────────────────────────────────────────────────────

def _inflate_ensemble(X: torch.Tensor, alpha: float) -> None:
    """
    Apply multiplicative covariance inflation in-place.

    Scales ensemble anomalies by √α, preserving the ensemble mean:

        X ← x̄ + √α · (X − x̄)

    This is equivalent to inflating the sample covariance by α:

        P_inflated = α · P_f

    Parameters
    ----------
    X:
        Ensemble state matrix (N, D), modified in-place.
    alpha:
        Inflation factor α ≥ 1.  α = 1.0 is a no-op.
    """
    if alpha == 1.0:
        return
    sqrt_alpha = math.sqrt(alpha)
    x_bar = X.mean(dim=0, keepdim=True)  # (1, D)
    X.sub_(x_bar)                         # X ← X − x̄
    X.mul_(sqrt_alpha)                    # X ← √α (X − x̄)
    X.add_(x_bar)                         # X ← x̄ + √α (X − x̄)
