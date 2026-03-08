"""
Abstract interface and shared Pydantic schemas for the PWR physics engine.

Any concrete solver (scipy, OpenMC wrapper, surrogate ML model, etc.) must
implement the ``ReactorSimulator`` abstract base class defined here so that
the rest of the application can rely on a stable, typed contract.

Physical model
--------------
Point-kinetics equations (PKE) with 6 delayed-neutron groups, coupled to a
two-node lumped thermal-hydraulic model:

  Neutronics
  ----------
  dn/dt   = [(ρ(t) − β) / Λ] · n(t) + Σᵢ λᵢ · Cᵢ(t)      [neutron pop.]
  dCᵢ/dt  = (βᵢ / Λ) · n(t) − λᵢ · Cᵢ(t)                  [precursors]

  Reactivity (linear feedback)
  ----------------------------
  ρ(t) = ρ_ext(t) + α_f · (T_f − T_f₀) + α_c · (T_c − T_c₀)

  Thermal-hydraulics (lumped, 2-node)
  ------------------------------------
  γ_f · dT_f/dt = P(t) − UA_fc · (T_f − T_c)
  γ_c · dT_c/dt = UA_fc · (T_f − T_c) − G_cool · (T_c − T_in)

  where P(t) = n(t) · P₀   (normalized neutron population × rated power).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from app.physics.constants import (
    BETA_GROUPS_U235,
    BETA_TOTAL_U235,
    COOLANT_FLOW_CAPACITY_W_PER_K,
    COOLANT_HEAT_CAPACITY_J_PER_K,
    COOLANT_INLET_TEMP_K,
    DOPPLER_COEFFICIENT,
    FUEL_COOLANT_CONDUCTANCE_W_PER_K,
    FUEL_HEAT_CAPACITY_J_PER_K,
    LAMBDA_GROUPS_U235,
    MODERATOR_COEFFICIENT,
    NOMINAL_COOLANT_TEMP_K,
    NOMINAL_FUEL_TEMP_K,
    NOMINAL_POWER_W,
    PROMPT_NEUTRON_LIFETIME_PWR,
)


class ReactorParams(BaseModel):
    """
    Complete parameter set for a single simulation run.

    All fields carry physical defaults derived from ``constants.py`` so that
    a caller can override only the parameters relevant to the scenario under
    study (e.g. ``external_reactivity`` for a step-insertion test).
    """

    # ── Neutron kinetics ─────────────────────────────────────────────────────
    prompt_neutron_lifetime: float = Field(
        default=PROMPT_NEUTRON_LIFETIME_PWR,
        gt=0,
        description="Effective prompt neutron generation time Λ [s]",
    )
    beta_groups: list[float] = Field(
        default=BETA_GROUPS_U235,
        min_length=6,
        max_length=6,
        description="Delayed-neutron fractions βᵢ for each of the 6 groups [-]",
    )
    lambda_groups: list[float] = Field(
        default=LAMBDA_GROUPS_U235,
        min_length=6,
        max_length=6,
        description="Decay constants λᵢ for each delayed-neutron group [s⁻¹]",
    )

    # ── Operating point ──────────────────────────────────────────────────────
    nominal_power: float = Field(
        default=NOMINAL_POWER_W,
        gt=0,
        description="Rated thermal power P₀ [W]",
    )
    coolant_inlet_temp: float = Field(
        default=COOLANT_INLET_TEMP_K,
        gt=0,
        description="Coolant inlet temperature T_in [K]",
    )
    nominal_fuel_temp: float = Field(
        default=NOMINAL_FUEL_TEMP_K,
        gt=0,
        description="Nominal average fuel temperature T_f₀ [K]",
    )
    nominal_coolant_temp: float = Field(
        default=NOMINAL_COOLANT_TEMP_K,
        gt=0,
        description="Nominal average coolant temperature T_c₀ [K]",
    )

    # ── Thermal-hydraulic parameters ─────────────────────────────────────────
    fuel_heat_capacity: float = Field(
        default=FUEL_HEAT_CAPACITY_J_PER_K,
        gt=0,
        description="Total fuel thermal mass γ_f = m_f · c_p,f [J/K]",
    )
    coolant_heat_capacity: float = Field(
        default=COOLANT_HEAT_CAPACITY_J_PER_K,
        gt=0,
        description="Total coolant thermal mass γ_c = m_c · c_p,c [J/K]",
    )
    fuel_coolant_conductance: float = Field(
        default=FUEL_COOLANT_CONDUCTANCE_W_PER_K,
        gt=0,
        description="Fuel-to-coolant overall heat-transfer conductance UA_fc [W/K]",
    )
    coolant_flow_capacity: float = Field(
        default=COOLANT_FLOW_CAPACITY_W_PER_K,
        gt=0,
        description="Coolant capacity flow rate G_cool = ṁ · c_p,c [W/K]",
    )

    # ── Reactivity feedback ──────────────────────────────────────────────────
    doppler_coefficient: float = Field(
        default=DOPPLER_COEFFICIENT,
        description="Fuel Doppler reactivity coefficient α_f [Δk/k per K] (negative for PWR)",
    )
    moderator_coefficient: float = Field(
        default=MODERATOR_COEFFICIENT,
        description="Moderator temperature reactivity coefficient α_c [Δk/k per K] (negative)",
    )

    # ── External / control perturbation ─────────────────────────────────────
    external_reactivity: float = Field(
        default=0.0,
        description=(
            "Step external reactivity insertion ρ_ext [Δk/k]. "
            "Positive → supercritical perturbation. 1 pcm = 1e-5."
        ),
    )

    @property
    def beta_total(self) -> float:
        """Total delayed-neutron fraction β = Σ βᵢ."""
        return sum(self.beta_groups)

    def verify_steady_state_consistency(self) -> None:
        """
        Raise ValueError if the nominal temperatures are inconsistent with
        the thermal parameters and nominal power.

        Checks:
          P₀ = UA_fc · (T_f₀ − T_c₀)
          P₀ = G_cool · (T_c₀ − T_in)
        """
        tol = 1e-3  # relative tolerance

        q_fc = self.fuel_coolant_conductance * (
            self.nominal_fuel_temp - self.nominal_coolant_temp
        )
        if abs(q_fc - self.nominal_power) / self.nominal_power > tol:
            raise ValueError(
                f"Steady-state inconsistency: UA_fc·(T_f0−T_c0) = {q_fc:.4e} W "
                f"≠ P₀ = {self.nominal_power:.4e} W"
            )

        q_cool = self.coolant_flow_capacity * (
            self.nominal_coolant_temp - self.coolant_inlet_temp
        )
        if abs(q_cool - self.nominal_power) / self.nominal_power > tol:
            raise ValueError(
                f"Steady-state inconsistency: G_cool·(T_c0−T_in) = {q_cool:.4e} W "
                f"≠ P₀ = {self.nominal_power:.4e} W"
            )


class SimulationResult(BaseModel):
    """
    Time-series output from a reactor simulation run.

    All arrays share the same length (number of output time points).
    The ``precursor_concentrations`` field has shape (n_times × 6).
    """

    time: list[float] = Field(description="Simulation time points [s]")
    neutron_population: list[float] = Field(
        description="Normalized neutron population n(t) [-]. n=1 at rated power."
    )
    power_W: list[float] = Field(description="Absolute thermal power P(t) = n(t)·P₀ [W]")
    fuel_temperature_K: list[float] = Field(description="Average fuel temperature T_f(t) [K]")
    coolant_temperature_K: list[float] = Field(
        description="Average coolant temperature T_c(t) [K]"
    )
    reactivity: list[float] = Field(
        description="Total core reactivity ρ(t) = ρ_ext + α_f·ΔT_f + α_c·ΔT_c [Δk/k]"
    )
    precursor_concentrations: list[list[float]] = Field(
        description="Delayed-neutron precursor group concentrations C_i(t), shape (n_times, 6)"
    )


class ReactorSimulator(ABC):
    """
    Abstract base class for PWR point-kinetics simulators.

    Concrete implementations (``ScipySolver``, future ML surrogates, etc.)
    must satisfy this interface so the rest of the application stays
    decoupled from the numerical backend.
    """

    @abstractmethod
    def run_simulation(
        self,
        params: ReactorParams,
        time_span: tuple[float, float],
        dt: float,
    ) -> SimulationResult:
        """
        Integrate the coupled PKE + thermal-hydraulic system.

        Parameters
        ----------
        params:
            Physical and operational parameters for the run.
        time_span:
            ``(t_start, t_end)`` integration interval [s].
        dt:
            Desired spacing of output evaluation points [s].
            The internal adaptive step size of the ODE solver is independent
            of this value and is controlled by error tolerances.

        Returns
        -------
        SimulationResult
            Complete time series of all state variables and derived quantities.

        Raises
        ------
        RuntimeError
            If the underlying ODE solver fails to converge.
        """
        ...
