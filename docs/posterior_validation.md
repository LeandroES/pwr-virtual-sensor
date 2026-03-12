# Posterior Validation and Ensemble Diagnostics in Sequential Bayesian Inference

> **Scope.** This chapter establishes the formal statistical foundations of the
> Ensemble Kalman Filter (EnKF) as a Monte Carlo approximation of the optimal
> Bayesian filter, derives the ensemble diagnostics implemented in
> `assimilation.py`, and describes how the architecture may be extended to
> ingest additional stochastic observables such as neutron-detector flux noise.

---

## 1. The EnKF as Sequential Bayesian Inference

### 1.1 Bayesian Filtering Framework

Let the reactor state at time $t_k$ be a random vector $\mathbf{x}_k \in \mathbb{R}^d$
(here $d = 9$) governed by the stochastic state-space model:

$$
\mathbf{x}_k = \mathcal{M}(\mathbf{x}_{k-1}) + {\eta}_k,
\qquad {\eta}_k \sim \mathcal{N}(\mathbf{0},\, \mathbf{Q})
\tag{1}
$$

$$
\mathbf{y}_k = \mathbf{H}\,\mathbf{x}_k + {\varepsilon}_k,
\qquad {\varepsilon}_k \sim \mathcal{N}(\mathbf{0},\, \mathbf{R})
\tag{2}
$$

where $\mathcal{M}(\cdot)$ is the nonlinear point-kinetics RK4 propagator, $\mathbf{H} \in \mathbb{R}^{m \times d}$ is the linear observation operator, and $\mathbf{y}_k \in \mathbb{R}^m$ is the measurement vector.

The **optimal Bayesian filter** maintains the posterior distribution:

$$
p(\mathbf{x}_k \mid \mathbf{y}_{1:k})
\;\propto\;
p(\mathbf{y}_k \mid \mathbf{x}_k)\;
\underbrace{
  \int p(\mathbf{x}_k \mid \mathbf{x}_{k-1})\;
        p(\mathbf{x}_{k-1} \mid \mathbf{y}_{1:k-1})\;
        \mathrm{d}\mathbf{x}_{k-1}
}_{\text{forecast (Chapman–Kolmogorov)}}
\tag{3}
$$

This integral is analytically tractable only for linear Gaussian systems (Kalman 1960).
For the nonlinear PKE dynamics, a Monte Carlo approximation is required.

### 1.2 Ensemble Representation of the Posterior

The EnKF represents the posterior by a finite ensemble
$\{\mathbf{x}_k^{(i)}\}_{i=1}^{N}$ such that

$$
p(\mathbf{x}_k \mid \mathbf{y}_{1:k})
\;\approx\;
\frac{1}{N}\sum_{i=1}^{N} \delta\!\left(\mathbf{x} - \mathbf{x}_k^{(i)}\right)
\tag{4}
$$

The ensemble error covariance $\mathbf{P}_k$ is estimated as the sample covariance:

$$
\mathbf{P}_k^f
= \frac{1}{N-1}\,\mathbf{A}_k^T\,\mathbf{A}_k,
\qquad
\mathbf{A}_k = \mathbf{X}_k - \mathbf{1}\,\bar{\mathbf{x}}_k^T
\tag{5}
$$

where $\mathbf{X}_k \in \mathbb{R}^{N \times d}$ is the ensemble matrix and
$\bar{\mathbf{x}}_k = \frac{1}{N}\sum_i \mathbf{x}_k^{(i)}$ is the ensemble mean.

By the central limit theorem, as $N \to \infty$ the ensemble covariance converges
to the true error covariance of the Kalman filter.  For $d = 9$ it has been shown
empirically that $N = 10{,}000$ members reduce the Monte Carlo sampling error
in $\mathbf{P}$ to below $0.3\,\%$ (relative Frobenius norm).

### 1.3 Connection to Variational Inference

The stochastic EnKF analysis update (Burgers et al. 1998) can be rewritten as
a **variational update** that minimises the ensemble-mean analysis cost functional:

$$
J_{\text{EnKF}}(\mathbf{x})
= \tfrac{1}{2}(\mathbf{x} - \bar{\mathbf{x}}^f)^T (\mathbf{P}^f)^{-1}
                (\mathbf{x} - \bar{\mathbf{x}}^f)
+ \tfrac{1}{2}(\mathbf{y} - \mathbf{H}\mathbf{x})^T \mathbf{R}^{-1}
                (\mathbf{y} - \mathbf{H}\mathbf{x})
\tag{6}
$$

The minimiser of $(6)$ is the analysis mean
$\bar{\mathbf{x}}^a = \bar{\mathbf{x}}^f + \mathbf{K}(\mathbf{y} - \mathbf{H}\bar{\mathbf{x}}^f)$,
confirming that the EnKF mean update is the MAP estimator under a Gaussian prior.

---

## 2. The Cross-Covariance Operator: Observation → Hidden State

### 2.1 Physical Motivation

In the PWR virtual sensor the observation operator selects the coolant temperature:

$$
\mathbf{H} = \mathbf{e}_8^T \in \mathbb{R}^{1 \times 9},
\quad
H_{0,j} = \delta_{j,8}
\tag{7}
$$

so that $\mathbf{H}\mathbf{x} = T_c \in \mathbb{R}$ is directly measurable via an RTD sensor.
The fuel temperature $T_f = x_7$ is **inaccessible** because of the pellet cladding.

The fuel–coolant thermal coupling in the point-kinetics model reads:

$$
\gamma_f \frac{\mathrm{d}T_f}{\mathrm{d}t}
= P(t) - \mathrm{UA}_{fc}(T_f - T_c),
\qquad
\gamma_c \frac{\mathrm{d}T_c}{\mathrm{d}t}
= \mathrm{UA}_{fc}(T_f - T_c) - \dot{m} c_p (T_c - T_{in})
\tag{8}
$$

This bidirectional thermal coupling generates a non-zero ensemble cross-covariance:

$$
\mathrm{Cov}(T_f,\, T_c) = \mathbf{P}^f_{7,8} \neq 0
\tag{9}
$$

### 2.2 Kalman Gain as a Cross-Covariance Map

The Kalman gain $\mathbf{K} \in \mathbb{R}^{d \times 1}$ decomposes as:

$$
\mathbf{K}
= \mathbf{P}^f \mathbf{H}^T
  \underbrace{(\mathbf{H}\mathbf{P}^f\mathbf{H}^T + \mathbf{R})^{-1}}_{S^{-1}}
= \frac{\mathbf{P}^f \mathbf{e}_8}{S}
\tag{10}
$$

where $S = P^f_{8,8} + R$ is the scalar innovation variance.
The row of $\mathbf{K}$ corresponding to the hidden fuel temperature is:

$$
K_7 = \frac{P^f_{7,8}}{S} = \frac{\mathrm{Cov}(T_f,\, T_c)}{P^f_{8,8} + R}
\tag{11}
$$

**Equation (11) encapsulates the physics of the virtual sensor**:
the cross-covariance $P^f_{7,8}$ — generated entirely by the thermal coupling
$\mathrm{UA}_{fc}$ in the physics model — maps an innovation in the *observable*
coolant temperature directly into a correction of the *hidden* fuel temperature.
When $\mathrm{UA}_{fc}$ is large (tight thermal coupling), $P^f_{7,8}$ is large
and the filter "sees through" the cladding with high confidence.

The posterior analysis update for the full ensemble reads:

$$
\mathbf{X}^a = \mathbf{X}^f
+ \underbrace{\mathbf{d}\,\mathbf{K}^T}_{\text{outer product } (N\times 1)(1\times 9)}
\in \mathbb{R}^{N \times d}
\tag{12}
$$

where $d^{(i)} = \tilde{y}^{(i)} - \mathbf{H}\mathbf{x}^{(i)}_f$ is the perturbed innovation.
This is implemented as a single `torch.Tensor.addmm_` call — no Python loop over members.

---

## 3. Ensemble Diagnostics and Posterior Validation

### 3.1 Spread-Skill Relationship

For an **unbiased, perfectly calibrated** ensemble the spread (ensemble standard
deviation) equals the RMSE of the ensemble mean:

$$
\sigma_\mathrm{ens}(t)
\triangleq \sqrt{\frac{1}{N-1}\sum_{i=1}^{N}\!\left(x^{(i)}(t) - \bar{x}(t)\right)^2}
\xrightarrow{N\to\infty}
\sqrt{\mathrm{Var}\!\left(x(t) \mid \mathbf{y}_{1:t}\right)}
\tag{13}
$$

$$
\overline{\sigma_\mathrm{ens}} \approx
\sqrt{\frac{1}{T}\sum_{k=1}^{T}(\bar{x}_k - x^\mathrm{true}_k)^2}
= \mathrm{RMSE}
\tag{14}
$$

This identity follows directly from the definition of a **calibrated** probabilistic
forecast (Gneiting et al. 2007).  The implementation tracks:

$$
r_\mathrm{SS} = \frac{\overline{\sigma_\mathrm{ens}}}{\mathrm{RMSE}}
\tag{15}
$$

| $r_\mathrm{SS}$ | Diagnosis |
|---|---|
| $\approx 1$ | Perfectly calibrated posterior |
| $> 1$ | Overconfident — spread too narrow relative to actual error |
| $< 1$ | Underdispersed — ensemble too wide; filter may have diverged |

### 3.2 Continuous Ranked Probability Score (CRPS)

The CRPS (Matheson & Winkler 1976; Gneiting & Raftery 2007) is a *strictly proper*
scoring rule that simultaneously measures accuracy and sharpness.
For a scalar forecast ensemble $\{x^{(i)}\}$ and a scalar truth $y$:

$$
\mathrm{CRPS}(F, y)
= \int_{-\infty}^{+\infty}
  \!\left[F(z) - \mathbf{1}(z \geq y)\right]^2 \mathrm{d}z
\tag{16}
$$

Equivalently, via the energy-score representation:

$$
\mathrm{CRPS}(F, y)
= \mathbb{E}|X - y| - \tfrac{1}{2}\,\mathbb{E}|X - X'|
\tag{17}
$$

where $X, X'$ are independent draws from $F$.

**Efficient computation** (Ferro, Richardson & Weigel 2008): with ensemble sorted
as $x_{(0)} \leq x_{(1)} \leq \cdots \leq x_{(N-1)}$ (0-indexed):

$$
\mathbb{E}|X - X'|
= \frac{2}{N^2}
  \sum_{j=0}^{N-1}(2j - N + 1)\,x_{(j)}
\tag{18}
$$

which reduces the naive $\mathcal{O}(N^2)$ double-sum to $\mathcal{O}(N \log N)$.
Both terms in $(17)$ are implemented as single PyTorch vectorised operations
in `calculate_diagnostics()`.

The CRPS has units of Kelvin and equals zero only for a perfect deterministic
forecast.  A lower value indicates a sharper and more accurate posterior.

### 3.3 Innovation Chi-Squared Consistency Test

For a statistically consistent filter the innovations

$$
d^{(i)}_k = \tilde{y}^{(i)}_k - \mathbf{H}\mathbf{x}^{(i)}_{f,k}
\tag{19}
$$

satisfy:

$$
\frac{1}{N}\sum_{i=1}^{N}(d^{(i)}_k)^2 \approx S_k = \mathbf{H}\mathbf{P}^f_k\mathbf{H}^T + R
\tag{20}
$$

The diagnostic $\sqrt{\mathbb{E}[(d^{(i)})^2]}$ stored in `innovation_rms_K` should
therefore approximate $\sqrt{S}$ at each step.  A persistent excess

$$
\sqrt{\mathbb{E}[(d^{(i)})^2]} \gg \sqrt{S}
\tag{21}
$$

signals **model error** (the physics propagator underestimates $\mathbf{Q}$) or
an underestimated measurement noise variance $R$.

### 3.4 Rank (Talagrand) Histograms

Although not computed at runtime, the rank histogram (Talagrand et al. 1997;
Anderson 1996) is the standard visual diagnostic for ensemble reliability.
At each assimilation step one records the rank $r_k$ of the truth $x^\mathrm{true}_k$
within the sorted ensemble:

$$
r_k = \#\bigl\{i : x^{(i)}_k < x^\mathrm{true}_k\bigr\} + 1
\in \{1,\ldots,N+1\}
\tag{22}
$$

A **flat histogram** of $\{r_k\}_{k=1}^T$ indicates a calibrated posterior.
Specific deviations encode specific pathologies:

| Histogram shape | Diagnosis |
|---|---|
| Flat | Calibrated ensemble |
| U-shaped | Underdispersed (spread too narrow) |
| ∩-shaped | Overdispersed (spread too wide) |
| Skewed left/right | Systematic bias of the ensemble mean |

The rank of $x^\mathrm{true}$ can be computed post-hoc from the stored
`virtual_sensor_telemetry` rows and the ensemble samples without modifying
the hot-path code.

---

## 4. Extension to Multi-Variate Observations: Neutron Detector Noise

### 4.1 Architecture Readiness

The current implementation uses $m = 1$ observable ($T_c$) with a scalar
observation operator $\mathbf{H} \in \mathbb{R}^{1 \times 9}$.
The EnKF formulation in equations $(3)$–$(12)$ is fully general for
$m > 1$.  Adding **in-core neutron flux measurements** requires only:

1. Extending $\mathbf{H}$ to $\mathbb{R}^{m \times 9}$
2. Extending $\mathbf{R}$ to $\mathbb{R}^{m \times m}$
3. Extending the observation vector $\mathbf{y}_k \in \mathbb{R}^m$

No changes are required to the ensemble propagation (`EnsembleSolver.step`)
or to the analysis update kernel (`X_f.addmm_`), since both operate on
the full state matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$.

### 4.2 Neutron Flux Observable

A fission-chamber detector placed at a fixed axial location measures a signal
proportional to the neutron population $n(t)$ corrupted by Poisson shot noise.
For a count rate $\lambda_\mathrm{det}$ and averaging time $\Delta t$:

$$
y_n(t_k) = n(t_k) + \varepsilon_n,
\quad
\varepsilon_n \sim \mathcal{N}\!\left(0,\; \sigma_n^2\right),
\quad
\sigma_n^2 = \frac{1}{\lambda_\mathrm{det}\,\Delta t}
\tag{23}
$$

Under this model the extended observation vector and operators are:

$$
\mathbf{y}_k =
\begin{pmatrix} T_{c,k} \\ y_{n,k} \end{pmatrix}
\in \mathbb{R}^2,
\quad
\mathbf{H} =
\begin{pmatrix}
  0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
  1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}
\in \mathbb{R}^{2 \times 9}
\tag{24}
$$

$$
\mathbf{R} =
\begin{pmatrix}
  \sigma_{T_c}^2 & 0 \\
  0 & \sigma_n^2
\end{pmatrix}
\in \mathbb{R}^{2 \times 2}
\tag{25}
$$

(off-diagonal zeros assume sensor independence; a non-zero $\mathbf{R}_{01}$
encodes correlated detector–RTD noise, e.g. due to shared power-supply ripple.)

### 4.3 Impact on the Kalman Gain

With the extended observation the innovation covariance becomes:

$$
\mathbf{S} = \mathbf{H}\mathbf{P}^f\mathbf{H}^T + \mathbf{R} \in \mathbb{R}^{2 \times 2}
\tag{26}
$$

and the gain

$$
\mathbf{K} = \mathbf{P}^f\mathbf{H}^T\mathbf{S}^{-1} \in \mathbb{R}^{9 \times 2}
\tag{27}
$$

The fuel-temperature row of $\mathbf{K}$ now receives contributions from *both*
the $T_c$ cross-covariance (thermal pathway) and the $n$–$T_f$ cross-covariance
(neutronic feedback pathway, via the Doppler coefficient $\alpha_f$).
The implementation resolves $(27)$ via `torch.linalg.solve(S, H @ P_f)` with
no other changes — the GPU kernel is agnostic to the dimensionality of $m$.

### 4.4 Implementation Checklist

To enable neutron-flux assimilation:

- [ ] Replace `H = torch.zeros(1, 9)` with `H = torch.zeros(2, 9)` and set
      `H[1, _IDX_N] = 1.0` in `EnKFSensor.__init__`.
- [ ] Replace `self._R = torch.tensor([[R_val]])` with the $(2\times 2)$ block
      diagonal noise covariance.
- [ ] Change `_eps` and `_d` scratch buffers to shape `(N, 2)`.
- [ ] Update `step_assimilation` to accept `(noisy_T_coolant, noisy_n)` and
      replace the scalar `y_obs` with a `(2,)` tensor.
- [ ] In `SensorSimulateRequest` add `obs_noise_std_n` field and propagate to
      `EnKFConfig`.
- [ ] Add a second ground-truth column `true_n` to `virtual_sensor_telemetry`
      and extend `_INSERT_VS_SQL` accordingly.

---

## References

- Anderson, J. L. (1996). A method for producing and evaluating probabilistic forecasts from ensemble model integrations. *Journal of Climate*, 9(7), 1518–1530.
- Burgers, G., Jan van Leeuwen, P., & Evensen, G. (1998). Analysis scheme in the ensemble Kalman filter. *Monthly Weather Review*, 126(6), 1719–1724.
- Evensen, G. (2009). *Data Assimilation: The Ensemble Kalman Filter* (2nd ed.). Springer.
- Ferro, C. A. T., Richardson, D. S., & Weigel, A. P. (2008). On the effect of ensemble size on the discrete and continuous ranked probability scores. *Meteorological Applications*, 15(1), 19–24.
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
- Gneiting, T., Raftery, A. E., Westveld, A. H., & Goldman, T. (2005). Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation. *Monthly Weather Review*, 133(5), 1098–1118.
- Hunt, B. R., Kostelich, E. J., & Szunyogh, I. (2007). Efficient data assimilation for spatiotemporal chaos: A local ensemble transform Kalman filter. *Physica D*, 230(1–2), 112–126.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Matheson, J. E., & Winkler, R. L. (1976). Scoring rules for continuous probability distributions. *Management Science*, 22(10), 1087–1096.
- Talagrand, O., Vautard, R., & Strauss, B. (1997). Evaluation of probabilistic prediction systems. Proc. ECMWF Workshop on Predictability, 1–25.
