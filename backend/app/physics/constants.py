"""
Standard nuclear data and nominal PWR operating parameters used throughout
the digital-twin physics engine.

Delayed-neutron group data for U-235 thermal fission (6-group Keepin model).
Reference: Keepin, G.R. (1965). Physics of Nuclear Kinetics. Addison-Wesley.
           IAEA-TECDOC-1234 (2001), Table IV.

All values are effective parameters for a thermal (PWR) neutron spectrum.
"""

# ── Delayed-neutron group fractions β_i (dimensionless) ─────────────────────
# Sum ≈ 0.006502 (650.2 pcm) — total delayed neutron fraction β for U-235
BETA_GROUPS_U235: list[float] = [
    2.15e-4,   # Group 1  — very long-lived  (T½ ≈ 55.7 s)
    1.424e-3,  # Group 2                     (T½ ≈ 22.7 s)
    1.274e-3,  # Group 3                     (T½ ≈  6.2 s)
    2.568e-3,  # Group 4                     (T½ ≈  2.3 s)
    7.48e-4,   # Group 5                     (T½ ≈  0.61 s)
    2.73e-4,   # Group 6  — very short-lived (T½ ≈  0.23 s)
]

# ── Delayed-neutron group decay constants λ_i [s⁻¹] ─────────────────────────
LAMBDA_GROUPS_U235: list[float] = [
    0.0124,   # Group 1
    0.0305,   # Group 2
    0.1110,   # Group 3
    0.3010,   # Group 4
    1.1400,   # Group 5
    3.0100,   # Group 6
]

#: Total effective delayed-neutron fraction β = Σ β_i
BETA_TOTAL_U235: float = sum(BETA_GROUPS_U235)  # ≈ 6.502e-3

# ── Prompt neutron generation time Λ [s] ─────────────────────────────────────
# For a thermal-spectrum, light-water-moderated PWR the prompt generation time
# is dominated by the thermalization time (~20 µs).
PROMPT_NEUTRON_LIFETIME_PWR: float = 2.0e-5  # [s]

# ── Nominal PWR operating point (representative 3 GWth four-loop plant) ──────
NOMINAL_POWER_W: float = 3_000e6         # Thermal power P₀          [W]
COOLANT_INLET_TEMP_K: float = 563.0      # T_in  = 290 °C             [K]
NOMINAL_FUEL_TEMP_K: float = 893.0       # T_f0  = 620 °C (avg fuel)  [K]
NOMINAL_COOLANT_TEMP_K: float = 593.0    # T_c0  = 320 °C (avg cool.) [K]

# ── Thermal-hydraulic lumped parameters ──────────────────────────────────────
#
# Two-node lumped model
#   Node 1 (fuel):    γ_f · dT_f/dt = P − UA_fc · (T_f − T_c)
#   Node 2 (coolant): γ_c · dT_c/dt = UA_fc · (T_f − T_c) − G_cool · (T_c − T_in)
#
# Derived from steady-state energy balances:
#   P₀ = UA_fc · (T_f0 − T_c0)  ⟹  UA_fc = 3e9 / 300 = 1e7 W/K
#   P₀ = G_cool · (T_c0 − T_in) ⟹  G_cool = 3e9 / 30  = 1e8 W/K
#
# Heat capacities chosen for realistic fuel/coolant thermal time constants:
#   τ_f = γ_f / UA_fc  ≈ 5 s   ⟹  γ_f = 5 × 10⁷ J/K
#       ↳ physical: m_f ≈ 167 t UO₂, c_p ≈ 300 J/(kg·K)
#   τ_c = γ_c / G_cool ≈ 5 s   ⟹  γ_c = 5 × 10⁸ J/K
#       ↳ physical: m_c ≈ 93 t H₂O (core), c_p ≈ 5 400 J/(kg·K)

FUEL_HEAT_CAPACITY_J_PER_K: float = 5.0e7   # γ_f  [J/K]
COOLANT_HEAT_CAPACITY_J_PER_K: float = 5.0e8  # γ_c  [J/K]
FUEL_COOLANT_CONDUCTANCE_W_PER_K: float = 1.0e7   # UA_fc [W/K]
COOLANT_FLOW_CAPACITY_W_PER_K: float = 1.0e8   # G_cool = ṁ·c_p [W/K]

# ── Reactivity feedback coefficients ─────────────────────────────────────────
# Both coefficients are negative → inherent safety (negative power coefficient).
DOPPLER_COEFFICIENT: float = -2.5e-5   # α_f  [Δk/k per K]  fuel Doppler
MODERATOR_COEFFICIENT: float = -2.0e-4  # α_c  [Δk/k per K]  moderator temp
