"""
Comprehensive test suite for the PWR point-kinetics + thermal-hydraulic engine.

Physics background
------------------
The coupled system has 9 state variables:
  n(t)     — normalized neutron population (n=1 at rated power P₀)
  Cᵢ(t)   — 6 delayed-precursor group concentrations
  T_f(t)  — average fuel temperature [K]
  T_c(t)  — average coolant temperature [K]

Analytical steady-state for a step ρ_ext insertion
---------------------------------------------------
At the new equilibrium all time-derivatives vanish, so total reactivity
must return to zero (criticality condition):

  ρ_total = ρ_ext + α_f·(T_f_ss − T_f₀) + α_c·(T_c_ss − T_c₀) = 0      (1)

Power balance at new SS:
  ΔP = P_ss − P₀
  T_f_ss − T_f₀ = ΔP / UA_fc                                             (2)
  T_c_ss − T_c₀ = ΔP / G_cool                                            (3)

Substituting (2)-(3) into (1):
  ΔP = −ρ_ext / (α_f/UA_fc + α_c/G_cool)                                 (4)

For the default PWR parameters and ρ_ext = 100 pcm = 1e-3:
  α_f/UA_fc  = −2.5e-5 / 1e7 = −2.5e-12  [1/W]
  α_c/G_cool = −2.0e-4 / 1e8 = −2.0e-12  [1/W]
  feedback_sens = −4.5e-12 [1/W]
  ΔP = −1e-3 / (−4.5e-12) ≈ 2.222×10⁸ W = 222.2 MW
  P_ss = 3000 + 222.2 = 3222.2 MW   →  n_ss ≈ 1.07407
  ΔT_f = 22.22 K                    →  T_f_ss ≈ 915.22 K
  ΔT_c =  2.22 K                    →  T_c_ss ≈ 595.22 K

These reference values are used throughout the tests.

Energy conservation identity
----------------------------
Integrating the sum of the two thermal ODEs:
  γ_f·dT_f/dt + γ_c·dT_c/dt = P(t) − G_cool·(T_c − T_in)

⟹  γ_f·[T_f(T) − T_f₀] + γ_c·[T_c(T) − T_c₀]
      = ∫₀ᵀ [P(t) − G_cool·(T_c(t) − T_in)] dt                          (5)

This identity is exact regardless of ρ_ext and provides a direct check on the
numerical accuracy of the ODE integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.physics.base import ReactorParams, SimulationResult
from app.physics.constants import (
    BETA_GROUPS_U235,
    LAMBDA_GROUPS_U235,
    PROMPT_NEUTRON_LIFETIME_PWR,
)
from app.physics.scipy_solver import ScipySolver

# ── Shared nominal parameter set ─────────────────────────────────────────────

NOMINAL = ReactorParams(
    prompt_neutron_lifetime=PROMPT_NEUTRON_LIFETIME_PWR,
    beta_groups=BETA_GROUPS_U235,
    lambda_groups=LAMBDA_GROUPS_U235,
    nominal_power=3_000e6,
    coolant_inlet_temp=563.0,
    nominal_fuel_temp=893.0,
    nominal_coolant_temp=593.0,
    fuel_heat_capacity=5.0e7,
    coolant_heat_capacity=5.0e8,
    fuel_coolant_conductance=1.0e7,
    coolant_flow_capacity=1.0e8,
    doppler_coefficient=-2.5e-5,
    moderator_coefficient=-2.0e-4,
    external_reactivity=0.0,
)

# ── Pre-computed analytical references for 100 pcm step ─────────────────────

RHO_EXT_100PCM: float = 100e-5  # 100 pcm = 1e-3 [Δk/k]

_FEEDBACK_SENS: float = (
    NOMINAL.doppler_coefficient / NOMINAL.fuel_coolant_conductance
    + NOMINAL.moderator_coefficient / NOMINAL.coolant_flow_capacity
)  # ≈ −4.5e-12 [1/W]

DELTA_P_ANALYTICAL: float = -RHO_EXT_100PCM / _FEEDBACK_SENS  # ≈ 2.222e8 W
P_SS_ANALYTICAL: float = NOMINAL.nominal_power + DELTA_P_ANALYTICAL  # ≈ 3.222e9 W
N_SS_ANALYTICAL: float = P_SS_ANALYTICAL / NOMINAL.nominal_power  # ≈ 1.07407

DELTA_TF_ANALYTICAL: float = DELTA_P_ANALYTICAL / NOMINAL.fuel_coolant_conductance   # ≈ 22.22 K
DELTA_TC_ANALYTICAL: float = DELTA_P_ANALYTICAL / NOMINAL.coolant_flow_capacity       # ≈  2.22 K

TF_SS_ANALYTICAL: float = NOMINAL.nominal_fuel_temp + DELTA_TF_ANALYTICAL
TC_SS_ANALYTICAL: float = NOMINAL.nominal_coolant_temp + DELTA_TC_ANALYTICAL


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def solver() -> ScipySolver:
    return ScipySolver()


@pytest.fixture(scope="module")
def perturbed_params() -> ReactorParams:
    return NOMINAL.model_copy(update={"external_reactivity": RHO_EXT_100PCM})


@pytest.fixture(scope="module")
def steady_result(solver: ScipySolver) -> SimulationResult:
    """100 s unperturbed run — used by steady-state tests."""
    return solver.run_simulation(NOMINAL, time_span=(0.0, 100.0), dt=1.0)


@pytest.fixture(scope="module")
def step_result(solver: ScipySolver, perturbed_params: ReactorParams) -> SimulationResult:
    """600 s run after 100 pcm insertion — used by transient tests."""
    return solver.run_simulation(perturbed_params, time_span=(0.0, 600.0), dt=1.0)


# ── Helper ────────────────────────────────────────────────────────────────────


def _late_mean(result: SimulationResult, arr_name: str, t_start: float = 500.0) -> float:
    """Average the named result array over the final quasi-steady-state window."""
    t = np.asarray(result.time)
    arr = np.asarray(getattr(result, arr_name))
    return float(np.mean(arr[t >= t_start]))


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — Steady-state (unperturbed) tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSteadyState:
    """Verify that the unperturbed critical reactor stays exactly at its
    nominal operating point for the full simulation window."""

    def test_neutron_population_constant(self, steady_result: SimulationResult) -> None:
        n = np.asarray(steady_result.neutron_population)
        max_dev = float(np.max(np.abs(n - 1.0)))
        assert max_dev < 1e-6, (
            f"Neutron population drifted by {max_dev:.2e} (expected < 1e-6)"
        )

    def test_power_constant_at_nominal(self, steady_result: SimulationResult) -> None:
        P = np.asarray(steady_result.power_W)
        assert np.allclose(P, NOMINAL.nominal_power, rtol=1e-6), (
            f"Power not constant: range [{P.min():.6e}, {P.max():.6e}] W"
        )

    def test_fuel_temperature_constant(self, steady_result: SimulationResult) -> None:
        T_f = np.asarray(steady_result.fuel_temperature_K)
        assert np.allclose(T_f, NOMINAL.nominal_fuel_temp, atol=1e-4), (
            f"Fuel temperature drifted: max |ΔT_f| = {np.max(np.abs(T_f - NOMINAL.nominal_fuel_temp)):.2e} K"
        )

    def test_coolant_temperature_constant(self, steady_result: SimulationResult) -> None:
        T_c = np.asarray(steady_result.coolant_temperature_K)
        assert np.allclose(T_c, NOMINAL.nominal_coolant_temp, atol=1e-4), (
            f"Coolant temperature drifted: max |ΔT_c| = {np.max(np.abs(T_c - NOMINAL.nominal_coolant_temp)):.2e} K"
        )

    def test_reactivity_zero_throughout(self, steady_result: SimulationResult) -> None:
        rho = np.asarray(steady_result.reactivity)
        assert np.allclose(rho, 0.0, atol=1e-9), (
            f"Non-zero reactivity at unperturbed SS: max|ρ| = {np.max(np.abs(rho)):.2e}"
        )

    def test_precursor_concentrations_constant(self, steady_result: SimulationResult) -> None:
        """Each Cᵢ(t) must remain equal to its initial equilibrium Cᵢ₀ = βᵢ/(λᵢ·Λ)."""
        beta = np.asarray(NOMINAL.beta_groups)
        lam = np.asarray(NOMINAL.lambda_groups)
        C_eq = beta / (lam * NOMINAL.prompt_neutron_lifetime)  # shape (6,)

        C = np.asarray(steady_result.precursor_concentrations)  # (n_times, 6)
        for g in range(6):
            max_dev = float(np.max(np.abs(C[:, g] - C_eq[g])))
            assert max_dev < 1e-6, (
                f"Group {g+1} precursor drifted by {max_dev:.2e} (C_eq = {C_eq[g]:.4f})"
            )

    def test_instantaneous_energy_balance(self, steady_result: SimulationResult) -> None:
        """
        At steady state, both heat-flux legs must exactly equal P₀:
          UA_fc·(T_f − T_c) = P₀   and   G_cool·(T_c − T_in) = P₀
        """
        P = np.asarray(steady_result.power_W)
        T_f = np.asarray(steady_result.fuel_temperature_K)
        T_c = np.asarray(steady_result.coolant_temperature_K)

        Q_fc = NOMINAL.fuel_coolant_conductance * (T_f - T_c)
        Q_out = NOMINAL.coolant_flow_capacity * (T_c - NOMINAL.coolant_inlet_temp)

        np.testing.assert_allclose(Q_fc, P, rtol=1e-5,
            err_msg="Fuel-to-coolant heat flux ≠ power at steady state")
        np.testing.assert_allclose(Q_out, P, rtol=1e-5,
            err_msg="Coolant-to-sink heat flux ≠ power at steady state")


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — Step-reactivity insertion tests (100 pcm)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStepReactivityInsertion:
    """
    Verify the physical response to a positive step of 100 pcm.
    The reactor is sub-prompt-critical (ρ_ext = 1e-3 < β ≈ 6.5e-3), so a
    supercritical excursion is expected, followed by stabilization due to
    negative thermal feedback.
    """

    def test_prompt_jump_within_first_second(self, step_result: SimulationResult) -> None:
        """
        The prompt-jump approximation predicts n_pj / n₀ = β/(β − ρ_ext).
        Power must be measurably above 1 within the very first output point.
        """
        n = np.asarray(step_result.neutron_population)
        beta_total = NOMINAL.beta_total
        n_pj_theoretical = beta_total / (beta_total - RHO_EXT_100PCM)  # ≈ 1.182

        # At t=1 s power should have at minimum started the prompt jump
        n_1s = n[1]
        assert n_1s > 1.0, (
            f"Power has not risen at t=1 s (n={n_1s:.5f}). Prompt jump not detected."
        )
        # But it should not exceed the theoretical prompt-jump ceiling by much
        assert n_1s <= n_pj_theoretical * 1.05, (
            f"Power at t=1 s ({n_1s:.4f}) greatly exceeds prompt-jump limit ({n_pj_theoretical:.4f})"
        )

    def test_power_peaks_before_feedback_arrests_rise(self, step_result: SimulationResult) -> None:
        """
        A clear power peak must exist.  After the peak, negative feedback
        brings the power back down toward the new steady-state level.
        """
        n = np.asarray(step_result.neutron_population)
        t = np.asarray(step_result.time)
        idx_peak = int(np.argmax(n))
        n_peak = float(n[idx_peak])
        t_peak = float(t[idx_peak])

        assert n_peak > N_SS_ANALYTICAL, (
            f"Peak power {n_peak:.5f} is not above the new SS level {N_SS_ANALYTICAL:.5f}. "
            "Overshoot (precursor buildup) expected."
        )
        assert t_peak < 300.0, (
            f"Power peak not reached within 300 s (t_peak = {t_peak:.1f} s)"
        )

        # After the peak, power must be declining
        n_post_peak = n[idx_peak + 5] if idx_peak + 5 < len(n) else n[-1]
        assert n_post_peak < n_peak, (
            "Power did not decline after the peak — negative feedback not acting."
        )

    def test_power_stabilizes_above_nominal(self, step_result: SimulationResult) -> None:
        """
        Final quasi-steady-state power must be above 1.0 (nominal) but
        the time-series in the last 100 s must be nearly flat (stable).
        """
        n = np.asarray(step_result.neutron_population)
        t = np.asarray(step_result.time)
        n_late = n[t >= 500.0]

        assert np.all(n_late > 1.0), (
            "Final power fell back to or below nominal — unphysical for positive ρ_ext."
        )
        sigma = float(np.std(n_late))
        assert sigma < 1e-4, (
            f"Power not yet stable in last 100 s: σ = {sigma:.2e} (expect < 1e-4)"
        )

    def test_new_steady_state_power_matches_analytical(self, step_result: SimulationResult) -> None:
        """
        New SS neutron population must match the analytical prediction from (4)
        to within 1 %.
        """
        n_final = _late_mean(step_result, "neutron_population")
        rel_err = abs(n_final - N_SS_ANALYTICAL) / N_SS_ANALYTICAL
        assert rel_err < 0.01, (
            f"n_ss: analytical = {N_SS_ANALYTICAL:.6f}, "
            f"simulated = {n_final:.6f}, "
            f"relative error = {rel_err:.3%}"
        )

    def test_new_fuel_temperature_matches_analytical(self, step_result: SimulationResult) -> None:
        """T_f_ss must match the value derived from (2): T_f₀ + ΔP/UA_fc."""
        T_f_final = _late_mean(step_result, "fuel_temperature_K")
        err_K = abs(T_f_final - TF_SS_ANALYTICAL)
        assert err_K < 1.0, (
            f"T_f_ss: analytical = {TF_SS_ANALYTICAL:.3f} K, "
            f"simulated = {T_f_final:.3f} K, "
            f"error = {err_K:.3f} K (tolerance = 1 K)"
        )

    def test_new_coolant_temperature_matches_analytical(self, step_result: SimulationResult) -> None:
        """T_c_ss must match the value derived from (3): T_c₀ + ΔP/G_cool."""
        T_c_final = _late_mean(step_result, "coolant_temperature_K")
        err_K = abs(T_c_final - TC_SS_ANALYTICAL)
        assert err_K < 1.0, (
            f"T_c_ss: analytical = {TC_SS_ANALYTICAL:.3f} K, "
            f"simulated = {T_c_final:.3f} K, "
            f"error = {err_K:.3f} K (tolerance = 1 K)"
        )

    def test_total_reactivity_zero_at_new_ss(self, step_result: SimulationResult) -> None:
        """
        Criticality condition: at new SS the total reactivity must vanish.
        ρ_ext is compensated by negative temperature feedback.
        """
        rho = np.asarray(step_result.reactivity)
        t = np.asarray(step_result.time)
        rho_late = rho[t >= 500.0]
        max_rho = float(np.max(np.abs(rho_late)))
        assert max_rho < 1e-4, (
            f"Total reactivity not zero at new SS: max|ρ| = {max_rho:.2e} [Δk/k] "
            f"(= {max_rho*1e5:.2f} pcm)"
        )

    def test_feedback_compensates_exactly(self, step_result: SimulationResult) -> None:
        """
        Direct check: ρ_ext + α_f·ΔT_f_ss + α_c·ΔT_c_ss ≈ 0.
        Uses simulation temperatures so this is independent of the analytical formula.
        """
        T_f_final = _late_mean(step_result, "fuel_temperature_K")
        T_c_final = _late_mean(step_result, "coolant_temperature_K")

        rho_feedback = (
            NOMINAL.doppler_coefficient * (T_f_final - NOMINAL.nominal_fuel_temp)
            + NOMINAL.moderator_coefficient * (T_c_final - NOMINAL.nominal_coolant_temp)
        )
        rho_total = RHO_EXT_100PCM + rho_feedback
        assert abs(rho_total) < 1e-4, (
            f"Reactivity balance at new SS: ρ_ext + ρ_fb = {rho_total:.2e} ≠ 0"
        )

    def test_both_temperatures_monotonically_increase_early(
        self, step_result: SimulationResult
    ) -> None:
        """
        Within the first 30 s both temperatures must rise — confirming that
        the inserted energy is heating the core before feedback arrests the transient.
        """
        t = np.asarray(step_result.time)
        T_f = np.asarray(step_result.fuel_temperature_K)
        T_c = np.asarray(step_result.coolant_temperature_K)

        idx_30 = int(np.searchsorted(t, 30.0))
        assert T_f[idx_30] > NOMINAL.nominal_fuel_temp + 1.0, (
            "Fuel temperature did not rise ≥ 1 K in first 30 s"
        )
        assert T_c[idx_30] > NOMINAL.nominal_coolant_temp, (
            "Coolant temperature did not rise in first 30 s"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — Energy conservation (macro balance)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnergyConservation:
    """
    Verify the integrated energy identity (5):

      γ_f·ΔT_f(T) + γ_c·ΔT_c(T)  =  ∫₀ᵀ [P(t) − G_cool·(T_c(t)−T_in)] dt

    This is an *exact algebraic consequence* of the thermal ODEs, so any
    violation indicates a numerical error in the solver.  With Radau at
    rtol=atol=1e-8 we expect the relative error to be well below 0.1 %.
    """

    def _net_energy(self, result: SimulationResult) -> tuple[float, float]:
        """
        Returns
        -------
        (energy_integral, energy_calorimetric)
          energy_integral     : ∫[P − G_cool·(T_c−T_in)] dt using trapezoidal rule
          energy_calorimetric : γ_f·ΔT_f + γ_c·ΔT_c  at end of simulation
        """
        t = np.asarray(result.time)
        P = np.asarray(result.power_W)
        T_c = np.asarray(result.coolant_temperature_K)
        T_f = np.asarray(result.fuel_temperature_K)

        G_cool = NOMINAL.coolant_flow_capacity
        T_in = NOMINAL.coolant_inlet_temp
        gamma_f = NOMINAL.fuel_heat_capacity
        gamma_c = NOMINAL.coolant_heat_capacity

        # Trapezoidal integration of [P(t) − G_cool·(T_c − T_in)]
        integrand = P - G_cool * (T_c - T_in)
        energy_integral = float(np.trapezoid(integrand, t))

        # Calorimetric estimate from final temperature state
        energy_calorimetric = float(
            gamma_f * (T_f[-1] - NOMINAL.nominal_fuel_temp)
            + gamma_c * (T_c[-1] - NOMINAL.nominal_coolant_temp)
        )
        return energy_integral, energy_calorimetric

    def test_energy_identity_unperturbed(self, steady_result: SimulationResult) -> None:
        """
        For the unperturbed SS both sides of (5) should be exactly zero
        (no stored energy, no net power imbalance).
        """
        E_int, E_cal = self._net_energy(steady_result)
        # Both are expected to be near zero; check their agreement
        assert abs(E_int) < 1.0, (
            f"Unperturbed net energy integral should be ≈ 0, got {E_int:.3e} J"
        )
        assert abs(E_cal) < 1.0, (
            f"Unperturbed calorimetric energy should be ≈ 0, got {E_cal:.3e} J"
        )

    def test_energy_identity_step_insertion(self, solver: ScipySolver) -> None:
        """
        After a 100 pcm step, the numerical integral must agree with the
        calorimetric estimate to within 0.5 %.

        The prompt jump raises power by ~18 % in < 100 ms.  A 1 s output
        step misses most of that area in the trapezoidal rule, so this test
        runs its own high-resolution simulation (dt = 0.01 s) which keeps the
        trapezoidal integration error well below the 0.5 % threshold.
        """
        params = NOMINAL.model_copy(update={"external_reactivity": RHO_EXT_100PCM})
        # Fine dt to capture the sub-second prompt-jump spike accurately
        hi_res = solver.run_simulation(params, time_span=(0.0, 600.0), dt=0.01)
        E_int, E_cal = self._net_energy(hi_res)

        assert E_int > 0, "Net stored energy must be positive for a power increase"
        assert E_cal > 0, "Temperature rise implies positive stored energy"

        rel_err = abs(E_int - E_cal) / abs(E_cal)
        assert rel_err < 0.005, (
            f"Energy conservation violated:\n"
            f"  ∫[P − Q_out] dt  = {E_int:.6e} J\n"
            f"  γ_f·ΔT_f + γ_c·ΔT_c = {E_cal:.6e} J\n"
            f"  relative error   = {rel_err:.4%} (tolerance 0.5 %)"
        )

    def test_stored_energy_matches_analytical_temperature_rise(
        self, step_result: SimulationResult
    ) -> None:
        """
        At new SS, the expected stored energy can be computed analytically from
        the temperature-rise predictions (2)-(3):
          ΔE_expected = γ_f·ΔT_f_anal + γ_c·ΔT_c_anal
        The calorimetric measurement from the simulation must agree within 1 %.
        """
        delta_E_analytical = (
            NOMINAL.fuel_heat_capacity * DELTA_TF_ANALYTICAL
            + NOMINAL.coolant_heat_capacity * DELTA_TC_ANALYTICAL
        )

        _, E_cal = self._net_energy(step_result)

        rel_err = abs(E_cal - delta_E_analytical) / delta_E_analytical
        assert rel_err < 0.01, (
            f"Stored energy:\n"
            f"  analytical = {delta_E_analytical:.4e} J\n"
            f"  simulated  = {E_cal:.4e} J\n"
            f"  relative error = {rel_err:.3%}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — Physical constraints (non-negativity, bounds)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhysicalConstraints:
    """
    State variables must obey hard physical limits at every time point.
    Violation indicates a badly-conditioned system or solver instability.
    """

    @pytest.fixture(scope="class")
    def constrained_result(self, solver: ScipySolver) -> SimulationResult:
        params = NOMINAL.model_copy(update={"external_reactivity": RHO_EXT_100PCM})
        return solver.run_simulation(params, time_span=(0.0, 300.0), dt=0.5)

    def test_neutron_population_non_negative(
        self, constrained_result: SimulationResult
    ) -> None:
        n = np.asarray(constrained_result.neutron_population)
        assert np.all(n >= 0.0), f"Neutron population went negative: min = {n.min():.4e}"

    def test_precursor_concentrations_non_negative(
        self, constrained_result: SimulationResult
    ) -> None:
        C = np.asarray(constrained_result.precursor_concentrations)  # (n_times, 6)
        assert np.all(C >= 0.0), (
            f"Precursor concentration went negative: min = {C.min():.4e} "
            f"(group {int(np.argmin(C.min(axis=0))) + 1})"
        )

    def test_coolant_above_inlet(self, constrained_result: SimulationResult) -> None:
        T_c = np.asarray(constrained_result.coolant_temperature_K)
        T_in = NOMINAL.coolant_inlet_temp
        assert np.all(T_c >= T_in), (
            f"Coolant temperature dropped below inlet ({T_in} K): min T_c = {T_c.min():.2f} K"
        )

    def test_fuel_above_coolant(self, constrained_result: SimulationResult) -> None:
        """Fuel must always be hotter than coolant when P > 0."""
        T_f = np.asarray(constrained_result.fuel_temperature_K)
        T_c = np.asarray(constrained_result.coolant_temperature_K)
        assert np.all(T_f >= T_c), (
            f"Fuel cooler than coolant: min (T_f−T_c) = {(T_f - T_c).min():.2f} K"
        )

    def test_power_bounded_below_prompt_criticality(
        self, constrained_result: SimulationResult
    ) -> None:
        """
        For sub-prompt-critical insertion (ρ_ext < β), power must not diverge.
        A loose upper bound of 3× nominal is imposed as a sanity check.
        """
        n = np.asarray(constrained_result.neutron_population)
        assert np.all(n < 3.0), (
            f"Power exceeded 3× nominal (unexpected divergence): max n = {n.max():.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — Interface and schema validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterfaceAndSchemas:
    """Unit-level tests for the abstract interface and Pydantic models."""

    def test_solver_is_reactor_simulator(self) -> None:
        from app.physics.base import ReactorSimulator
        assert isinstance(ScipySolver(), ReactorSimulator)

    def test_result_has_correct_length(self, steady_result: SimulationResult) -> None:
        n = len(steady_result.time)
        assert len(steady_result.neutron_population) == n
        assert len(steady_result.power_W) == n
        assert len(steady_result.fuel_temperature_K) == n
        assert len(steady_result.coolant_temperature_K) == n
        assert len(steady_result.reactivity) == n
        assert len(steady_result.precursor_concentrations) == n

    def test_precursor_shape(self, steady_result: SimulationResult) -> None:
        """Each time-point entry must have exactly 6 group values."""
        for row in steady_result.precursor_concentrations:
            assert len(row) == 6

    def test_steady_state_consistency_validator(self) -> None:
        """verify_steady_state_consistency must pass for the nominal params."""
        NOMINAL.verify_steady_state_consistency()  # must not raise

    def test_steady_state_consistency_validator_detects_error(self) -> None:
        bad_params = NOMINAL.model_copy(update={"nominal_fuel_temp": 999.0})
        with pytest.raises(ValueError, match="Steady-state inconsistency"):
            bad_params.verify_steady_state_consistency()

    def test_invalid_time_span_raises(self, solver: ScipySolver) -> None:
        with pytest.raises(ValueError, match="t_end > t_start"):
            solver.run_simulation(NOMINAL, time_span=(100.0, 0.0), dt=1.0)

    def test_invalid_dt_raises(self, solver: ScipySolver) -> None:
        with pytest.raises(ValueError, match="dt"):
            solver.run_simulation(NOMINAL, time_span=(0.0, 10.0), dt=0.0)

    def test_beta_total_property(self) -> None:
        expected = sum(BETA_GROUPS_U235)
        assert abs(NOMINAL.beta_total - expected) < 1e-12

    def test_power_equals_n_times_p0(self, step_result: SimulationResult) -> None:
        n = np.asarray(step_result.neutron_population)
        P = np.asarray(step_result.power_W)
        np.testing.assert_allclose(P, n * NOMINAL.nominal_power, rtol=1e-10)
