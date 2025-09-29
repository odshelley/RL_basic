# Behavioral SAC (Choquet/CPT) — Build Spec for Code Agents

**Objective:** Extend a standard Soft Actor-Critic (SAC) agent (continuous actions, MuJoCo) with rank-dependent / CPT evaluation, optional β–δ discounting, and optional hyperbolic discounting via exponential mixtures. Keep SAC’s policy update unchanged; replace critic targets with distorted expectations.

---

## Theory Requirements (for correctness)

1. **Distorted expectation (Choquet integral).**
   - For random variable X with CDF F, distortion g:[0,1]→[0,1] increasing with g(0)=0, g(1)=1:
     - ρ_g(X) = ∫_0^1 F^{-1}(u) d g(u).
   - Discrete estimator (values x_(1)≤…≤x_(K), C_i=i/K): weights π^(i)=g(C_i)-g(C_{i-1}); ρ_g ≈ Σ_i π^(i) x_(i).
   - Required properties: **monotone** and **translation-invariant**; ρ_g is **1-Lipschitz in ∥·∥_∞** (independent of g).

2. **CPT operator (loss aversion + reference).**
   - Value functions u_+, u_- : R_+→R_+ **increasing and Lipschitz** with constants L_+, L_-; loss aversion λ≥1; reference b(s).
   - CPT functional for next-state values V(S′):
     - Ψ(V | s,a) = ρ_{g+}( u_+((V(S′)−b(s))_+) ) − ρ_{g-}( λ·u_-((b(s)−V(S′))_+) ).
   - CPT optimality operator: (T_CPT V)(s) = max_a { r(s,a) + γ·Ψ(V | s,a) }.
   - **Contraction:** if b is exogenous, ∥T_CPT V − T_CPT W∥_∞ ≤ γ(L_+ + λ L_-) ∥V−W∥_∞. If b=b(V), multiply by (1+L_b).

3. **Soft improvement (SAC-style) still holds.**
   - For Boltzmann policy π*(·|s) ∝ exp(Q(s,·)/α), the soft state functional F_π(s)=E[Q−α log π] satisfies F_{π*}(s)−F_π(s)=α·KL(π||π*)≥0.
   - Since ρ_g and Ψ are **monotone**, this improvement carries through the distorted evaluation. Actor update remains standard SAC.

4. **Discounting modes.**
   - Standard: γ∈(0,1).
   - **β–δ (quasi-hyperbolic):** use γ_β = β·δ in one-step backups (stationary contraction).
   - **Hyperbolic:** Γ(k)=1/(1+κk). Implement as safe **finite mixture of exponentials**: Γ(k)≈Σ_i p_i γ_i^k (Σ p_i=1).

5. **About Prelec function (probability weighting, not utility).**
   - g_Prelec(p) = exp(−η (−ln p)^α), with α>0, η>0.
   - It is **strictly increasing** on (0,1) ⇒ valid as distortion.
   - It is **not globally Lipschitz** on (0,1) for typical α∈(0,1); derivative can blow up near 0 or 1. This is **OK**: the theory requires u_± to be Lipschitz; ρ_g itself is 1-Lipschitz regardless of g. For numerics, clip p∈[ε,1−ε].

---

## Implementation Tasks

### 0) Config Schema (YAML)
```yaml
behavioral:
  mode: "none|choquet|cpt"
  estimator: "scalar|iqn"
  # probability weighting
  g_type: "identity|prelec|wang|custom"
  g_params: { alpha: 0.65, eta: 1.0 }  # for Prelec
  # CPT
  lambda_loss_aversion: 2.0
  u_plus:   { type: "power", alpha: 0.88, eps: 1e-6 }   # u(x)=max(x,0)^alpha
  u_minus:  { type: "power", alpha: 0.88, eps: 1e-6 }   # applied to |x|
  reference:
    type: "constant|state_value|ema_return"
    constant: 0.0
    ema_tau: 0.01
  # distributional (IQN) options
  iqn:
    n_quantiles: 64
    clip_quantiles: [1e-3, 1-1e-3]
  # discounting
  discounting:
    type: "standard|beta_delta|hyperbolic_mixture"
    gamma: 0.99
    beta: 1.0
    delta: 0.99
    mixture: { gammas: [0.99, 0.95, 0.90], probs: [0.6, 0.3, 0.1] }
```

### 1) Distortion Utilities
- Implement `DistortionFunction` with:
  - `__call__(p)` returns g(p).
  - `inv(u)` returns generalized inverse g^{-1}(u) (monotone).
  - Built-ins: identity; **Prelec** (alpha, eta); **Wang** (z-score shift); custom callable.

- Implement `choquet_expectation(values: Tensor[K]) -> Tensor[1]`:
  - Sort ascending, compute cumulative C_i=i/K, weights π^(i)=g(C_i)−g(C_{i−1}), return Σ π^(i) v_(i).

- Implement CPT wrapper:
  ```python
  def cpt_functional(values, ref, u_plus, u_minus, g_plus, g_minus, lam):
      gains  = torch.clamp(values - ref, min=0.0)
      losses = torch.clamp(ref - values, min=0.0)
      val_g  = u_plus(gains)
      val_l  = u_minus(losses)
      cho_g  = choquet_expectation(val_g)
      cho_l  = choquet_expectation(val_l)
      return cho_g - lam * cho_l
  ```
  - u_plus/u_minus must be **increasing** and **Lipschitz** (e.g., `u(x)=((x+eps)**alpha - eps**alpha)` with 0<alpha≤1 ensures Lipschitz near 0).

### 2) Scalar-Critic Path (minimal SAC changes)
- Keep twin critics Q^1_θ, Q^2_θ and target copies.
- For each transition (s,a,r,s′,d):
  1) Sample M actions a′_m ~ π(·|s′).
  2) Compute soft values v_m = min_j Q_{\bar θ}^j(s′,a′_m) − α log π(a′_m|s′).
  3) Distorted backup:
     - **Choquet:** y = r + γ* · ρ_g({v_m}).
     - **CPT:** y = r + γ* · CPT({v_m}, ref=b(s)).
     - γ* from discounting mode (standard/β–δ/mixture). For mixture, either apply ρ once then scale by Σ p_i γ_i, or compute per-component and sum (document choice; recommend per-component sum).
  4) Critic loss: standard MSE to y (stop-grad on y).
- Actor update: unchanged SAC objective using current critics.
- Temperature α: keep auto-tuning.

### 3) Distributional (IQN) Path
- Replace critics with IQN heads Z_θ(s,a;u).
- To compute ρ_g of soft next-state value:
  1) Draw T_m ~ U[0,1], set U_m = g^{-1}(T_m).
  2) For each U_m, evaluate Z_{\bar θ}^{soft}(s′; U_m).
  3) Average over m: (1/M) Σ_m Z(·;U_m).
- CPT: split quantile samples around reference b(s), apply u_+, u_-, g_+, g_-; multiply loss side by λ; aggregate as in CPT wrapper.
- Quantile loss: quantile Huber / QR loss to distorted target distribution (or scalar target).

### 4) Reference b(s)
- Implement providers:
  - `constant(c=0.0)`.
  - `ema_return(tau)` (episode-level EMA; detach from gradient).
  - `state_value_baseline` (stop-grad estimate of expected return).
- Default: constant(0.0).

### 5) Discounting
- Standard γ.
- β–δ: use γ_β = β·δ in one-step target.
- Hyperbolic mixture: supply arrays `gammas`, `probs`; compute `Σ_i p_i γ_i * distorted_value_i`.

### 6) Diagnostics & Safety
- Log: critic target mean/std; Choquet weight entropy; fraction of “loss” samples; λ; discount mode; α.
- Warn if approximate small-gain margin violated: γ*(L_+ + λ L_-) (× (1+L_b) if b depends on V) ≥ 1.
- Unit tests:
  - Identity g, λ=1, u_±(x)=x ⇒ reduces to vanilla SAC (numerical equality within tolerance).
  - Synthetic distributions: numeric Choquet/CPT vs. analytic values.
  - Inverse-sampling check: Monte Carlo via U=g^{-1}(T) matches Riemann–Stieltjes sum.

### 7) Default Hyperparameters
- Choquet only: Prelec(α=0.65, η=1.0); M=8 action samples for scalar path.
- CPT: λ=2.0; u_±(x)=x^{0.88}; b=0.0.
- IQN: n_quantiles=64; clip U∈[1e−3, 1−1e−3].
- Slightly lower critic LR than vanilla SAC to offset heavier tails.

### 8) Repository Structure
```
behavioral/
  distortions.py      # g, inv_g, choquet, CPT utilities
  reference.py        # reference providers
  discounting.py      # standard, beta-delta, mixture
agents/sac/
  behavioral_sac.py   # critic target swap + flags
  iqn_heads.py        # optional distributional critics
configs/
  mujoco_behavioral.yaml
tests/
  test_choquet.py
  test_cpt.py
  test_inverse_sampling.py
```

---

## API Expectations for the Agent

- Input: existing SAC codebase with Gaussian policy, twin critics, replay buffer.
- Output: a feature flag `--behavioral.mode` enabling Choquet/CPT; YAML config; fully integrated training loop changes; unit tests pass.
- Environments: MuJoCo continuous-control (Pendulum-v1, Hopper, Walker2d, etc.).
- No changes to environment interfaces.

---

## Notes on Prelec Monotonicity and Lipschitzness

- Prelec g is **strictly increasing** on (0,1) ⇒ valid.
- g is **not globally Lipschitz** near 0 or 1 for common α∈(0,1); use quantile clipping (ε≈1e−3).
- The contraction conditions in CPT depend on **Lipschitz of u_±** and reference dynamics (L_b), **not** on Lipschitz of g. The Choquet operator ρ_g is **1-Lipschitz** in sup-norm for any increasing g with g(0)=0, g(1)=1.