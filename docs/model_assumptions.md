# PWR Digital Twin — Physical Model Assumptions

> **Scope**: This document describes the mathematical model implemented in
> `backend/app/physics/` (Point Kinetics Equations + lumped thermal-hydraulics).
> It covers governing equations, parameter provenance, numerical method, and
> explicit limitations.

---

## 1. Governing Equations

### 1.1 Point Kinetics Equations (PKE)

The neutron population is represented by a single scalar `n(t)` (normalized so
that `n = 1` equals rated power P₀).  The full spatial neutron flux is
replaced by a time-dependent amplitude multiplied by a fixed fundamental mode
shape — the "point kinetics" approximation.

```
dn/dt  = [(ρ(t) − β) / Λ] · n(t) + Σᵢ λᵢ · Cᵢ(t)       (i = 1 … 6)

dCᵢ/dt = (βᵢ / Λ) · n(t) − λᵢ · Cᵢ(t)
```

| Symbol | Definition | Units |
|--------|-----------|-------|
| `n(t)` | Normalized neutron population | — |
| `ρ(t)` | Total core reactivity | Δk/k |
| `β` | Total delayed-neutron fraction = Σ βᵢ | Δk/k |
| `βᵢ` | Delayed-neutron fraction for group i | Δk/k |
| `Λ` | Effective prompt-neutron generation time | s |
| `λᵢ` | Decay constant of delayed-neutron group i | s⁻¹ |
| `Cᵢ(t)` | Precursor concentration for group i | — |

---

### 1.2 Reactivity Model (Linear Feedback)

Reactivity is the sum of an external (control) perturbation and first-order
temperature feedbacks:

```
ρ(t) = ρ_ext + α_f · (T_f(t) − T_f₀) + α_c · (T_c(t) − T_c₀)
```

| Symbol | Definition | Units |
|--------|-----------|-------|
| `ρ_ext` | External reactivity (step insertion) | Δk/k |
| `α_f` | Fuel Doppler coefficient (negative) | Δk/k per K |
| `α_c` | Moderator temperature coefficient (negative) | Δk/k per K |
| `T_f(t)` | Average fuel temperature | K |
| `T_c(t)` | Average coolant temperature | K |
| `T_f₀`, `T_c₀` | Nominal temperatures at rated power | K |

Both coefficients are negative, providing inherent passive safety (negative
power coefficient).

---

### 1.3 Thermal-Hydraulic Model (Two-Node Lumped)

The core is represented by two thermal nodes: a single fuel node and a single
coolant node.  Heat transfer between them and from coolant to the sink is
modelled by conductance/capacity-rate terms.

```
γ_f · dT_f/dt = P(t) − UA_fc · (T_f − T_c)

γ_c · dT_c/dt = UA_fc · (T_f − T_c) − G_cool · (T_c − T_in)

P(t) = n(t) · P₀
```

| Symbol | Definition | Units |
|--------|-----------|-------|
| `γ_f` | Total fuel thermal mass (m_f · c_p,f) | J/K |
| `γ_c` | Total coolant thermal mass (m_c · c_p,c) | J/K |
| `UA_fc` | Fuel-to-coolant overall conductance | W/K |
| `G_cool` | Coolant capacity flow rate (ṁ · c_p,c) | W/K |
| `T_in` | Coolant inlet temperature | K |
| `P₀` | Rated thermal power | W |

**Steady-state energy balances** (derivation of `UA_fc` and `G_cool`):

```
P₀ = UA_fc · (T_f₀ − T_c₀)   →   UA_fc = 3×10⁹ / 300 = 1×10⁷ W/K
P₀ = G_cool · (T_c₀ − T_in)  →   G_cool = 3×10⁹ / 30  = 1×10⁸ W/K
```

---

## 2. Parameter Values

### 2.1 Six-Group Delayed-Neutron Data (U-235 Thermal Fission)

Source: Keepin (1965) *Physics of Nuclear Kinetics*; IAEA-TECDOC-1234 (2001).
These are effective parameters for a thermal (LWR) neutron spectrum.

| Group | βᵢ (Δk/k) | λᵢ (s⁻¹) | Half-life T½ (s) | Physical precursor nuclides |
|------:|----------:|----------:|----------------:|----------------------------|
| 1 | 2.150×10⁻⁴ | 0.0124 | 55.9 | ⁸⁷Br, ¹⁴²Cs |
| 2 | 1.424×10⁻³ | 0.0305 | 22.7 | ¹³⁷I, ⁸⁸Br, ⁸⁹Br |
| 3 | 1.274×10⁻³ | 0.1110 |  6.2 | ¹³⁸Cs, ⁸⁹Kr, ⁹³Rb |
| 4 | 2.568×10⁻³ | 0.3010 |  2.3 | ¹³⁹Xe, ⁹³Kr, ⁹⁴Rb |
| 5 | 7.480×10⁻⁴ | 1.1400 |  0.61 | ¹⁴⁰Xe, ¹⁴⁵Cs, ⁹⁴Kr |
| 6 | 2.730×10⁻⁴ | 3.0100 |  0.23 | ¹⁴¹Xe, ¹⁴⁵Cs, ⁸⁴As |

**β_total** = Σ βᵢ = **6.502×10⁻³** (650.2 pcm)

**Prompt-neutron generation time** Λ = **2.0×10⁻⁵ s** (20 µs), representative
of a light-water-moderated thermal reactor dominated by slowing-down time.

---

### 2.2 Nominal PWR Operating Point (representative 3 GWth four-loop plant)

| Parameter | Symbol | Value | Units |
|-----------|--------|------:|-------|
| Rated thermal power | P₀ | 3 000 × 10⁶ | W |
| Coolant inlet temperature | T_in | 563.0 | K (290 °C) |
| Nominal avg fuel temperature | T_f₀ | 893.0 | K (620 °C) |
| Nominal avg coolant temperature | T_c₀ | 593.0 | K (320 °C) |

---

### 2.3 Lumped Thermal-Hydraulic Parameters

| Parameter | Symbol | Value | Units | Physical basis |
|-----------|--------|------:|-------|----------------|
| Fuel thermal mass | γ_f | 5.0×10⁷ | J/K | ≈ 167 t UO₂, c_p ≈ 300 J/(kg·K) |
| Coolant thermal mass | γ_c | 5.0×10⁸ | J/K | ≈ 93 t H₂O (core), c_p ≈ 5 400 J/(kg·K) |
| Fuel-coolant conductance | UA_fc | 1.0×10⁷ | W/K | Derived from P₀/(T_f₀−T_c₀) |
| Coolant capacity flow rate | G_cool | 1.0×10⁸ | W/K | Derived from P₀/(T_c₀−T_in) |

Implied thermal time constants: τ_f = γ_f/UA_fc ≈ **5 s**,
τ_c = γ_c/G_cool ≈ **5 s**.

---

### 2.4 Reactivity Feedback Coefficients

| Coefficient | Symbol | Value | Units | Sign convention |
|-------------|--------|------:|-------|-----------------|
| Fuel Doppler | α_f | −2.5×10⁻⁵ | Δk/k per K | Negative → stable |
| Moderator temperature | α_c | −2.0×10⁻⁴ | Δk/k per K | Negative → stable |

Combined feedback sensitivity (power coefficient):

```
dρ/dP|_SS = α_f/UA_fc + α_c/G_cool ≈ −4.5×10⁻¹² Δk/k per W  (< 0 → stable)
```

---

### 2.5 New Steady State After External Step Reactivity Insertion

For a step insertion ρ_ext at t = 0, the new thermal steady state is:

```
ΔP_SS = −ρ_ext / (α_f/UA_fc + α_c/G_cool)

e.g., ρ_ext = 100 pcm = 1×10⁻³  →  ΔP_SS ≈ +222 MW  →  n_SS ≈ 1.074
```

---

## 3. Numerical Method

### 3.1 ODE System Stiffness

The coupled (9-variable) system is stiff.  The stiffness ratio is set by the
shortest time scale (prompt neutrons, ~20 µs) versus the longest (delayed
group 1, ~56 s and thermal, ~5 s):

```
Stiffness ratio ≈ τ_slow / τ_prompt ≈ 56 / 2×10⁻⁵ ≈ 3×10⁶
```

The dominant eigenvalue of the prompt neutron subsystem at criticality is
roughly `s_0 ≈ −β/Λ ≈ −325 s⁻¹`.  Explicit methods (RK45) would require
`dt < 3 ms` for stability, making 100 s transients impractical.

### 3.2 Primary Solver (ScipySolver)

`scipy.integrate.solve_ivp` with `method='Radau'` (5th-order, L-stable
implicit Runge-Kutta).  Error tolerances: `rtol = atol = 1×10⁻⁸`.

- **Adaptive step size**: controlled internally by Radau; output is evaluated
  at user-specified `t_eval` points.
- **Dense output disabled**: the solver interpolates only to the requested
  output grid, minimizing memory.
- **Thread safety**: `ScipySolver` carries no mutable state between calls.

### 3.3 Batch/Tensor Solver (TensorSolver)

Fixed-step classical RK4 implemented with vectorized array operations (NumPy
by default, PyTorch/JAX optionally via `PHYSICS_BACKEND` env var).

- **Step-size constraint for stability**: `dt ≤ 0.001 s` recommended.
  At `dt = 0.001 s`, `dt · |s_0| ≈ 0.33`, well within the RK4 stability
  region (|z| ≤ 2.83 along the negative real axis).
- **Batch dimension**: processes N independent parameter sets simultaneously
  with state shape `(N, 9)` — suitable for parameter sweeps / UQ.
- **Not adaptive**: accuracy degrades for large dt; use `ScipySolver` for
  production results requiring `rtol/atol` control.

---

## 4. Model Limitations

The following simplifications are made relative to a production-grade nuclear
safety code (RELAP5, TRACE, SIMULATE-3, etc.).  Each limitation is noted for
awareness and future roadmap planning.

### 4.1 Spatial Homogenization (Point Kinetics)

The neutron flux is assumed to follow a fixed spatial mode shape at all times.
This is only valid when:
- Power excursions are mild (< ~20 % amplitude).
- Reactivity insertions are symmetric across the core.
- No control rod ejection or asymmetric-loading transients.

**Impact**: Spatial xenon oscillations, asymmetric rod withdrawal, and large
LOCAs cannot be accurately modelled.

### 4.2 Linear Reactivity Feedback

Both α_f and α_c are constant (first-order Taylor expansion about T_f₀ and
T_c₀).  In reality:
- The Doppler coefficient α_f depends on fuel temperature as ~1/√T and becomes
  less negative at high burnup.
- The moderator temperature coefficient varies strongly with boron
  concentration, void fraction, and fuel burnup.

**Impact**: Feedback is underestimated at large temperature excursions
(T_f ≫ T_f₀) and overestimated in the presence of boron dilution.

### 4.3 Lumped Two-Node Thermal Model

The entire fuel and coolant masses are each represented as single uniform-
temperature nodes.  Actual cores have radial and axial temperature
distributions with significant gradients:
- Peak fuel centerline temperatures in a PWR reach ~1 200–1 500 °C vs.
  an average of ~620 °C modelled here.
- Axial coolant temperature rises ~30–35 K from inlet to outlet; only the
  average is tracked.

**Impact**: Doppler feedback is underestimated (should be weighted by local
flux and fuel temperature); departure from nucleate boiling (DNBR) cannot be
assessed.

### 4.4 Single-Phase Coolant (No Void Fraction)

The model assumes liquid-phase coolant at all times.  There is no steam
generation, no two-phase flow, and no critical heat flux calculation.

**Impact**: Scenarios involving coolant boiling (anticipated operational
occurrences at high power, or loss-of-flow accidents) cannot be simulated.

### 4.5 No Fission-Product Poisoning (Xenon / Samarium)

¹³⁵Xe and ¹⁴⁹Sm absorb neutrons and cause substantial reactivity changes on
hour-long time scales.  Neither is modelled.

**Impact**: Power-shaping operations, xenon oscillations, and post-shutdown
xenon peak cannot be studied.  The model is valid only for transients shorter
than ~30 minutes.

### 4.6 No Control-Rod Dynamics

Control-rod position, worth curves, and differential/integral rod worth are
not modelled.  Reactivity is inserted as an instantaneous step `ρ_ext` with no
ramp or partial-insertion effects.

**Impact**: Realistic control-rod-withdrawal accidents or reactor trips cannot
be accurately simulated.

### 4.7 No Fuel Burnup

Fuel composition (U-235, Pu-239 build-in, FP accumulation) is constant.
β_total and α_f drift by ~20 % over a typical 18-month fuel cycle.

**Impact**: Kinetics parameters are representative of beginning-of-cycle (BOC)
only.

### 4.8 Constant Inlet Temperature and Flow Rate

T_in and G_cool are fixed at their nominal values.  Primary pump rundown,
pressurizer transients, and secondary-side steam generator dynamics are not
modelled.

**Impact**: Loss-of-flow accidents and feed/bleed operations are outside scope.

### 4.9 Six-Group Keepin Parameters are Spectrum-Averaged

The β_i and λ_i values are the classic 1965 Keepin data for U-235 in a thermal
spectrum.  Modern evaluations (e.g., JEFF-3.3, ENDF/B-VIII) differ by 1–3 %.
Mixed U-235/Pu-239 fuels (typical of equilibrium cores) have lower β_total
(~500–520 pcm) and slightly different group structures.

**Impact**: A ~20 % overestimate of β at equilibrium burnup leads to a
correspondingly overestimated prompt-critical margin.

---

## 5. Validity Envelope

The model is suited for:

- **Reactivity perturbations** up to ±200 pcm with linear thermal feedback.
- **Transient durations** of seconds to ~30 minutes (before xenon effects).
- **Teaching and prototyping** of reactor dynamics control algorithms.
- **Parameter sensitivity studies** (batch tensor solver) for uncertainty
  quantification of kinetics coefficients.

It is **not suited** for:

- Accident analysis requiring spatially resolved neutronics.
- Loss-of-coolant accidents (LOCA).
- Post-trip long-term decay heat (no decay heat model).
- Licensing or safety-case work.

---

*Last updated: 2026-03-07 — Fase 5 (Documentación e Infraestructura GPU)*
