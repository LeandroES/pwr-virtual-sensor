"""
Concrete implementation of ReactorSimulator using scipy.integrate.solve_ivp
with the Radau implicit Runge-Kutta method.

Why Radau / BDF?
----------------
The coupled PKE system is stiff: the prompt-neutron time scale (~20 µs) is
six orders of magnitude shorter than the slowest delayed-group decay constant
(~80 s) and the thermal time constants (~5 s).  Explicit methods (RK45, etc.)
require a step size bounded by the shortest time scale, making them impractical
for transients of hundreds of seconds.  Radau is L-stable and excels at such
systems.

State vector layout (9 components)
------------------------------------
  y[0]     n       normalized neutron population [-]  (n = 1 → rated power P₀)
  y[1..6]  C_i     delayed-precursor group concentrations (i = 1 … 6)
  y[7]     T_f     average fuel temperature [K]
  y[8]     T_c     average coolant temperature [K]

Governing equations
-------------------
Point kinetics (Keepin, 1965):
  dn/dt   = [(ρ − β) / Λ] · n + Σᵢ λᵢ · Cᵢ
  dCᵢ/dt  = (βᵢ / Λ) · n − λᵢ · Cᵢ

Reactivity (linear, first-order feedback):
  ρ(t) = ρ_ext + α_f · (T_f − T_f₀) + α_c · (T_c − T_c₀)

Thermal hydraulics (two-node lumped):
  γ_f · dT_f/dt = P(t) − UA_fc · (T_f − T_c)
  γ_c · dT_c/dt = UA_fc · (T_f − T_c) − G_cool · (T_c − T_in)
  P(t) = n(t) · P₀
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from app.physics.base import ReactorParams, ReactorSimulator, SimulationResult


class ScipySolver(ReactorSimulator):
    """
    Stiff ODE solver for the coupled PWR point-kinetics / thermal-hydraulic
    system.  Uses ``scipy.integrate.solve_ivp`` with ``method='Radau'``.

    The solver is stateless: every call to ``run_simulation`` is independent
    and thread-safe (no shared mutable state).
    """

    # ODE solver error tolerances — tight enough for <0.01 % energy error
    _RTOL: float = 1e-8
    _ATOL: float = 1e-8

    # ── Public API ────────────────────────────────────────────────────────────

    def run_simulation(
        self,
        params: ReactorParams,
        time_span: tuple[float, float],
        dt: float,
    ) -> SimulationResult:
        """
        Integrate the PKE + thermal system and return full state time series.

        Parameters
        ----------
        params:
            Reactor physical and operating parameters.
        time_span:
            ``(t_start, t_end)`` [s].  Must satisfy t_end > t_start.
        dt:
            Output time-step spacing [s].  The internal adaptive step used
            by Radau is controlled by ``_RTOL`` / ``_ATOL`` independently.

        Returns
        -------
        SimulationResult

        Raises
        ------
        ValueError
            If ``time_span`` or ``dt`` are inconsistent.
        RuntimeError
            If the Radau solver fails (e.g. due to singular Jacobian or
            step-size underflow).
        """
        t0, tf = time_span
        if tf <= t0:
            raise ValueError(f"time_span must satisfy t_end > t_start, got {time_span}")
        if dt <= 0 or dt > (tf - t0):
            raise ValueError(f"dt={dt} is incompatible with time_span={time_span}")

        y0 = self._initial_state(params)
        rhs = self._build_rhs(params)
        t_eval = np.arange(t0, tf + dt * 0.5, dt)  # inclusive of tf when divisible
        t_eval = t_eval[t_eval <= tf]

        sol = solve_ivp(
            fun=rhs,
            t_span=(t0, tf),
            y0=y0,
            method="Radau",
            t_eval=t_eval,
            rtol=self._RTOL,
            atol=self._ATOL,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(
                f"Radau solver failed after {sol.nfev} function evaluations. "
                f"Message: {sol.message}"
            )

        return self._build_result(sol.t, sol.y, params)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _initial_state(self, params: ReactorParams) -> NDArray[np.float64]:
        """
        Build the critical steady-state initial condition vector y₀.

        At criticality (ρ = 0, n = 1) the precursor balance gives:
            dCᵢ/dt = 0  ⟹  Cᵢ₀ = βᵢ / (λᵢ · Λ)

        Verification:
            Σᵢ λᵢ · Cᵢ₀ = Σᵢ βᵢ / Λ = β / Λ
            dn/dt|₀ = [(0 − β)/Λ]·1 + β/Λ = 0  ✓

        Temperatures are set to their nominal values defined in ``params``.
        The thermal equations are also trivially zero at t = 0 because
        P₀ = UA_fc · (T_f₀ − T_c₀) = G_cool · (T_c₀ − T_in).
        """
        beta = np.asarray(params.beta_groups, dtype=np.float64)
        lam = np.asarray(params.lambda_groups, dtype=np.float64)
        Lambda = params.prompt_neutron_lifetime

        y0: NDArray[np.float64] = np.empty(9, dtype=np.float64)
        y0[0] = 1.0                        # n₀  — normalized, critical
        y0[1:7] = beta / (lam * Lambda)    # Cᵢ₀ — precursor equilibrium
        y0[7] = params.nominal_fuel_temp   # T_f₀
        y0[8] = params.nominal_coolant_temp  # T_c₀
        return y0

    def _build_rhs(
        self,
        params: ReactorParams,
    ) -> Callable[[float, NDArray[np.float64]], NDArray[np.float64]]:
        """
        Return the RHS function f(t, y) expected by ``solve_ivp``.

        All parameters are captured by closure so the inner function is a
        plain Python callable with minimal overhead.
        """
        # Pre-extract and convert to numpy arrays once — avoids repeated
        # attribute lookups and list→array conversions inside the hot loop.
        beta: NDArray[np.float64] = np.asarray(params.beta_groups, dtype=np.float64)
        lam: NDArray[np.float64] = np.asarray(params.lambda_groups, dtype=np.float64)
        beta_total: float = float(np.sum(beta))
        Lambda: float = params.prompt_neutron_lifetime

        P0: float = params.nominal_power
        T_f0: float = params.nominal_fuel_temp
        T_c0: float = params.nominal_coolant_temp
        T_in: float = params.coolant_inlet_temp

        gamma_f: float = params.fuel_heat_capacity
        gamma_c: float = params.coolant_heat_capacity
        UA_fc: float = params.fuel_coolant_conductance
        G_cool: float = params.coolant_flow_capacity

        alpha_f: float = params.doppler_coefficient
        alpha_c: float = params.moderator_coefficient
        rho_ext: float = params.external_reactivity

        # Precompute the factor β / Λ used in every precursor equation
        beta_over_Lambda: NDArray[np.float64] = beta / Lambda

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            n = y[0]
            C = y[1:7]
            T_f = y[7]
            T_c = y[8]

            # ── Reactivity (linear feedback) ─────────────────────────────
            rho = rho_ext + alpha_f * (T_f - T_f0) + alpha_c * (T_c - T_c0)

            # ── Point kinetics ────────────────────────────────────────────
            # dn/dt = [(ρ − β)/Λ]·n + Σ λᵢ·Cᵢ
            dn_dt = ((rho - beta_total) / Lambda) * n + np.dot(lam, C)

            # dCᵢ/dt = (βᵢ/Λ)·n − λᵢ·Cᵢ
            dC_dt = beta_over_Lambda * n - lam * C

            # ── Thermal power ─────────────────────────────────────────────
            P = n * P0

            # ── Thermal-hydraulic (two-node lumped) ───────────────────────
            # Q_fc = heat flux from fuel to coolant [W]
            Q_fc = UA_fc * (T_f - T_c)

            # γ_f · dT_f/dt = P − Q_fc
            dTf_dt = (P - Q_fc) / gamma_f

            # γ_c · dT_c/dt = Q_fc − G_cool·(T_c − T_in)
            dTc_dt = (Q_fc - G_cool * (T_c - T_in)) / gamma_c

            dy: NDArray[np.float64] = np.empty(9, dtype=np.float64)
            dy[0] = dn_dt
            dy[1:7] = dC_dt
            dy[7] = dTf_dt
            dy[8] = dTc_dt
            return dy

        return rhs

    def _build_result(
        self,
        t: NDArray[np.float64],
        y: NDArray[np.float64],
        params: ReactorParams,
    ) -> SimulationResult:
        """
        Post-process the raw solver output into a ``SimulationResult``.

        Parameters
        ----------
        t:  shape (n_times,)
        y:  shape (9, n_times)
        """
        n_arr = y[0]               # (n_times,)
        C_arr = y[1:7]             # (6, n_times)
        T_f_arr = y[7]             # (n_times,)
        T_c_arr = y[8]             # (n_times,)

        # Recompute total reactivity at every output point
        rho_arr = (
            params.external_reactivity
            + params.doppler_coefficient * (T_f_arr - params.nominal_fuel_temp)
            + params.moderator_coefficient * (T_c_arr - params.nominal_coolant_temp)
        )

        return SimulationResult(
            time=t.tolist(),
            neutron_population=n_arr.tolist(),
            power_W=(n_arr * params.nominal_power).tolist(),
            fuel_temperature_K=T_f_arr.tolist(),
            coolant_temperature_K=T_c_arr.tolist(),
            reactivity=rho_arr.tolist(),
            # Transpose: (6, n_times) → list of (n_times,) lists → (n_times, 6)
            precursor_concentrations=C_arr.T.tolist(),
        )
