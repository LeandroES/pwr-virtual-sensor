"""
Vectorized batch solver for the PWR Point Kinetics + Thermal-Hydraulic system.

Purpose
-------
This module implements the *same physics* as ``ScipySolver`` but using pure
array operations instead of the adaptive Radau integrator.  The design goal is
throughput on many simultaneous parameter sets (uncertainty quantification,
parameter sweeps, surrogate training), not per-run accuracy.

Backend selection (``PHYSICS_BACKEND`` env var)
----------------------------------------------
+-----------+------------------------------------------------------------+
| Value     | Array module used                                          |
+===========+============================================================+
| ``numpy`` | NumPy — always available, CPU only.  Default.              |
+-----------+------------------------------------------------------------+
| ``torch`` | PyTorch — GPU via ROCm (AMD) or CUDA (NVIDIA) if           |
|           | available; CPU fallback otherwise.                         |
+-----------+------------------------------------------------------------+

The public API (``run_simulation``, ``run_batch``) always returns plain Python
lists / ``SimulationResult`` objects regardless of backend, so the rest of the
application never needs to know which array module was active.

Numerical method: Classical RK4 (fixed step)
--------------------------------------------
Classical 4th-order Runge-Kutta is used instead of an adaptive stiff solver.
This makes the step size the user's responsibility.

Stability constraint for the PKE stiff eigenvalue:

    |dt · s₀| ≤ 2.83        (RK4 stability region on negative real axis)
    |s₀| ≈ β/Λ = 6.5×10⁻³ / 2×10⁻⁵ = 325 s⁻¹

    ⟹  dt_max ≈ 2.83 / 325 ≈ 8.7 ms

``dt = 0.001 s`` (1 ms) is recommended.  Values above ``0.01 s`` raise a
``ValueError`` to prevent silent instability.

CPU fallback guarantee
----------------------
``tensor_solver.py`` imports ``torch`` *lazily* (only when
``PHYSICS_BACKEND=torch``).  If torch is not installed the module still loads
and works perfectly with NumPy.  The CPU NumPy path is always available.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from app.physics.base import ReactorParams, ReactorSimulator, SimulationResult

# ── Maximum safe fixed step size ─────────────────────────────────────────────
_DT_MAX: float = 0.01  # [s] — beyond this explicit RK4 may go unstable

# ── Backend selection ─────────────────────────────────────────────────────────

class _ArrayModule(Protocol):
    """Minimal numpy-like protocol used by the solver internals."""

    def array(self, data: Any, dtype: Any = None) -> Any: ...
    def zeros(self, shape: Any, dtype: Any = None) -> Any: ...
    def empty(self, shape: Any, dtype: Any = None) -> Any: ...
    def stack(self, arrays: Any, axis: int = 0) -> Any: ...


def _load_backend() -> tuple[Any, str]:
    """
    Return ``(array_module, backend_name)`` according to ``PHYSICS_BACKEND``.

    Falls back to NumPy if the requested backend is unavailable.
    """
    requested = os.environ.get("PHYSICS_BACKEND", "numpy").lower().strip()

    if requested == "torch":
        try:
            import torch  # noqa: PLC0415

            device_name = "cpu"
            if torch.cuda.is_available():
                # Works for both NVIDIA CUDA and AMD ROCm (HIP)
                device_name = "cuda"
            elif hasattr(torch, "hip") and torch.hip.is_available():  # type: ignore[attr-defined]
                device_name = "hip"

            # Return a thin adapter that gives numpy-compatible semantics for
            # the subset of operations used in this module.
            return _TorchAdapter(torch, device_name), f"torch[{device_name}]"
        except ImportError:
            pass  # fall through to numpy

    # Default: NumPy (CPU)
    return np, "numpy"


class _TorchAdapter:
    """
    Wraps a subset of the torch API to match the numpy signatures used here.

    Only the operations actually called by ``TensorSolver`` are implemented.
    This is intentionally minimal — not a general torch↔numpy bridge.
    """

    def __init__(self, torch: Any, device: str) -> None:
        self._torch = torch
        self._device = device

    # dtype constants (mimic numpy)
    @property
    def float64(self) -> Any:
        return self._torch.float64

    def array(self, data: Any, dtype: Any = None) -> Any:
        dt = dtype if dtype is not None else self._torch.float64
        return self._torch.tensor(data, dtype=dt, device=self._device)

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        dt = dtype if dtype is not None else self._torch.float64
        if isinstance(shape, int):
            shape = (shape,)
        return self._torch.zeros(shape, dtype=dt, device=self._device)

    def empty(self, shape: Any, dtype: Any = None) -> Any:
        dt = dtype if dtype is not None else self._torch.float64
        if isinstance(shape, int):
            shape = (shape,)
        return self._torch.empty(shape, dtype=dt, device=self._device)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return self._torch.stack(arrays, dim=axis)

    def tolist(self, tensor: Any) -> list:
        return tensor.cpu().tolist()

    def to_numpy(self, tensor: Any) -> NDArray[np.float64]:
        return tensor.cpu().numpy()


# ── Vectorized RHS ────────────────────────────────────────────────────────────

def _build_batch_rhs(params_list: Sequence[ReactorParams], xp: Any):
    """
    Build a batched RHS function  f(y) → dy  where shapes are (N, 9).

    Parameters are stacked into 1-D arrays of length N so that all N
    simulations advance together with a single set of array operations.

    Returned function signature:  ``rhs(y: array(N,9)) -> array(N,9)``
    (no explicit ``t`` argument — the model has no explicit time dependence
    other than through the state; ρ_ext is a step applied at t=0).
    """
    N = len(params_list)
    f64 = xp.float64 if hasattr(xp, "float64") else np.float64

    def _col(values: list[float]) -> Any:
        return xp.array(values, dtype=f64)  # shape (N,) or (N, 6)

    # ── Kinetics parameters (all shape (N,) or (N,6)) ────────────────────
    beta_mat = xp.array(
        [p.beta_groups for p in params_list], dtype=f64
    )  # (N, 6)
    lam_mat = xp.array(
        [p.lambda_groups for p in params_list], dtype=f64
    )  # (N, 6)
    beta_total = _col([sum(p.beta_groups) for p in params_list])   # (N,)
    Lambda = _col([p.prompt_neutron_lifetime for p in params_list]) # (N,)

    # ── Operating point ───────────────────────────────────────────────────
    P0 = _col([p.nominal_power for p in params_list])                # (N,)
    T_f0 = _col([p.nominal_fuel_temp for p in params_list])          # (N,)
    T_c0 = _col([p.nominal_coolant_temp for p in params_list])       # (N,)
    T_in = _col([p.coolant_inlet_temp for p in params_list])         # (N,)

    # ── Thermal-hydraulic constants ───────────────────────────────────────
    gamma_f = _col([p.fuel_heat_capacity for p in params_list])       # (N,)
    gamma_c = _col([p.coolant_heat_capacity for p in params_list])    # (N,)
    UA_fc = _col([p.fuel_coolant_conductance for p in params_list])   # (N,)
    G_cool = _col([p.coolant_flow_capacity for p in params_list])     # (N,)

    # ── Reactivity feedback ───────────────────────────────────────────────
    alpha_f = _col([p.doppler_coefficient for p in params_list])      # (N,)
    alpha_c = _col([p.moderator_coefficient for p in params_list])    # (N,)
    rho_ext = _col([p.external_reactivity for p in params_list])      # (N,)

    # Precomputed β/Λ term for each group and batch element: (N, 6)
    beta_over_Lambda = beta_mat / Lambda[:, None]  # type: ignore[index]

    def rhs(y: Any) -> Any:
        """
        Evaluate dy/dt for all N batch elements simultaneously.

        y : array of shape (N, 9)
            Columns: [n, C1, C2, C3, C4, C5, C6, T_f, T_c]
        returns dy : array of shape (N, 9)
        """
        n   = y[:, 0]       # (N,)
        C   = y[:, 1:7]     # (N, 6)
        T_f = y[:, 7]       # (N,)
        T_c = y[:, 8]       # (N,)

        # ── Reactivity ────────────────────────────────────────────────
        rho = rho_ext + alpha_f * (T_f - T_f0) + alpha_c * (T_c - T_c0)

        # ── Neutron population ────────────────────────────────────────
        # Σᵢ λᵢ·Cᵢ  for each batch element: (N, 6) · (N, 6) → sum → (N,)
        lambda_C_sum = (lam_mat * C).sum(axis=-1)  # type: ignore[call-overload]

        dn_dt = ((rho - beta_total) / Lambda) * n + lambda_C_sum

        # ── Precursor groups ──────────────────────────────────────────
        # (N, 6) = (N, 6) * (N,)[:,None]  −  (N, 6) * (N, 6)
        dC_dt = beta_over_Lambda * n[:, None] - lam_mat * C  # type: ignore[index]

        # ── Thermal power ─────────────────────────────────────────────
        P = n * P0   # (N,)

        # ── Heat fluxes ───────────────────────────────────────────────
        Q_fc = UA_fc * (T_f - T_c)                                    # (N,)
        dTf_dt = (P - Q_fc) / gamma_f                                 # (N,)
        dTc_dt = (Q_fc - G_cool * (T_c - T_in)) / gamma_c            # (N,)

        # ── Assemble dy ───────────────────────────────────────────────
        # Stack column-wise: (N,) pieces → (N, 9)
        dy = xp.stack(
            [
                dn_dt,
                dC_dt[:, 0], dC_dt[:, 1], dC_dt[:, 2],  # type: ignore[index]
                dC_dt[:, 3], dC_dt[:, 4], dC_dt[:, 5],  # type: ignore[index]
                dTf_dt,
                dTc_dt,
            ],
            axis=-1,
        )
        return dy

    return rhs


# ── Initial state helper ──────────────────────────────────────────────────────

def _build_initial_state(params_list: Sequence[ReactorParams], xp: Any) -> Any:
    """
    Build y₀ of shape (N, 9) from a list of ReactorParams.

    At criticality (ρ=0, n=1):
        Cᵢ₀ = βᵢ / (λᵢ · Λ)
    """
    f64 = xp.float64 if hasattr(xp, "float64") else np.float64
    N = len(params_list)

    rows: list[list[float]] = []
    for p in params_list:
        beta = np.asarray(p.beta_groups)
        lam  = np.asarray(p.lambda_groups)
        C0   = (beta / (lam * p.prompt_neutron_lifetime)).tolist()
        row  = [1.0] + C0 + [p.nominal_fuel_temp, p.nominal_coolant_temp]
        rows.append(row)

    return xp.array(rows, dtype=f64)  # (N, 9)


# ── RK4 step ──────────────────────────────────────────────────────────────────

def _rk4_step(rhs: Any, y: Any, dt: float) -> Any:
    """Single classical RK4 step: y(t+dt) from y(t)."""
    k1 = rhs(y)
    k2 = rhs(y + (dt * 0.5) * k1)
    k3 = rhs(y + (dt * 0.5) * k2)
    k4 = rhs(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ── Public solver class ───────────────────────────────────────────────────────

class TensorSolver(ReactorSimulator):
    """
    Fixed-step RK4 solver with a vectorized batch dimension.

    **Single run** — compatible with the ``ReactorSimulator`` ABC:
        ``result = TensorSolver().run_simulation(params, (0, 100), dt=0.001)``

    **Batch run** — N parameter sets at once:
        ``results = TensorSolver().run_batch([p1, p2, ...], (0, 100), dt=0.001)``

    Backend is chosen once at instantiation from the ``PHYSICS_BACKEND``
    environment variable.  CPU (NumPy) is always the fallback.
    """

    def __init__(self) -> None:
        self._xp, self._backend_name = _load_backend()

    @property
    def backend(self) -> str:
        """Name of the active array backend (e.g. ``'numpy'``, ``'torch[cuda]'``)."""
        return self._backend_name

    # ── ReactorSimulator ABC ──────────────────────────────────────────────────

    def run_simulation(
        self,
        params: ReactorParams,
        time_span: tuple[float, float],
        dt: float,
    ) -> SimulationResult:
        """
        Single-parameter simulation.  Delegates to ``run_batch`` with N=1.

        Parameters
        ----------
        params:
            Reactor parameters.
        time_span:
            ``(t_start, t_end)`` [s].
        dt:
            Fixed integration / output step [s].  Must be ≤ 0.01 s.

        Returns
        -------
        SimulationResult
        """
        return self.run_batch([params], time_span, dt)[0]

    # ── Batch API ─────────────────────────────────────────────────────────────

    def run_batch(
        self,
        params_list: Sequence[ReactorParams],
        time_span: tuple[float, float],
        dt: float,
    ) -> list[SimulationResult]:
        """
        Simulate N independent reactor configurations simultaneously.

        All N configurations share the same ``time_span`` and ``dt`` but can
        have completely different physical parameters.

        Parameters
        ----------
        params_list:
            Sequence of ``ReactorParams`` objects.  Length N ≥ 1.
        time_span:
            ``(t_start, t_end)`` [s].
        dt:
            Fixed integration step [s].  Must satisfy dt ≤ 0.01 for RK4
            stability with PWR-typical stiffness (|s₀| ≈ 325 s⁻¹).

        Returns
        -------
        list[SimulationResult]
            One ``SimulationResult`` per input parameter set, in order.

        Raises
        ------
        ValueError
            If ``dt`` exceeds the stability threshold, or ``time_span`` is
            inconsistent.
        """
        t0, tf = time_span
        if tf <= t0:
            raise ValueError(f"time_span must satisfy t_end > t_start, got {time_span}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if dt > _DT_MAX:
            raise ValueError(
                f"dt={dt} s exceeds the RK4 stability limit of {_DT_MAX} s for "
                f"the stiff PWR PKE system.  Use dt ≤ {_DT_MAX} s or switch to "
                f"ScipySolver for adaptive step-size control."
            )
        if not params_list:
            return []

        xp = self._xp
        N = len(params_list)

        # ── Build batch state and RHS ─────────────────────────────────────
        y = _build_initial_state(params_list, xp)  # (N, 9)
        rhs = _build_batch_rhs(params_list, xp)

        # ── Time grid ─────────────────────────────────────────────────────
        t_points_np = np.arange(t0, tf + dt * 0.5, dt)
        t_points_np = t_points_np[t_points_np <= tf]
        n_steps = len(t_points_np)

        # Pre-allocate output: (n_steps, N, 9)
        history_np = np.empty((n_steps, N, 9), dtype=np.float64)

        # ── Integration loop ──────────────────────────────────────────────
        # Record initial state (t0), then advance
        history_np[0] = self._to_numpy(y)

        for step_idx in range(1, n_steps):
            y = _rk4_step(rhs, y, dt)
            history_np[step_idx] = self._to_numpy(y)

        # ── Pack results ──────────────────────────────────────────────────
        results: list[SimulationResult] = []
        for i, p in enumerate(params_list):
            traj = history_np[:, i, :]   # (n_steps, 9)
            n_arr  = traj[:, 0]
            C_mat  = traj[:, 1:7]        # (n_steps, 6)
            T_f    = traj[:, 7]
            T_c    = traj[:, 8]

            rho = (
                p.external_reactivity
                + p.doppler_coefficient   * (T_f - p.nominal_fuel_temp)
                + p.moderator_coefficient * (T_c - p.nominal_coolant_temp)
            )

            results.append(
                SimulationResult(
                    time=t_points_np.tolist(),
                    neutron_population=n_arr.tolist(),
                    power_W=(n_arr * p.nominal_power).tolist(),
                    fuel_temperature_K=T_f.tolist(),
                    coolant_temperature_K=T_c.tolist(),
                    reactivity=rho.tolist(),
                    precursor_concentrations=C_mat.tolist(),
                )
            )

        return results

    # ── Internal helper ───────────────────────────────────────────────────────

    def _to_numpy(self, arr: Any) -> NDArray[np.float64]:
        """Convert backend tensor to a NumPy array for output packing."""
        if isinstance(arr, np.ndarray):
            return arr
        # PyTorch tensor
        return arr.detach().cpu().numpy()
