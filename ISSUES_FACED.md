1. Check clip_actions in IsaacLabVecEnvWrapper.
2. Observation space issue.
3. Adaptive learning rate (PPO issue).
4. Bootstrapping on time outs (observation space).

5. KL Divergence issue. 
Replace Schulman’s KL surrogate with full score-function estimator
Schulman’s squared-log-ratio Taylor approximation under-penalizes large policy shifts
and drops the score-function term, causing biased ∇KL and trust-region leaks.
Now use the unbiased sequence-level estimator (or control-variate variants)
to restore exact KL gradients, enforce the PPO trust region, and speed up convergence.


# KL Regularisation in PPO – Why the Exact KL Beats the Surrogate

Several popular PPO forks swap the exact forward KL with **Schulman’s low-variance surrogate**.  
That shortcut lowers variance but **biases the gradient** and can silently break the trust-region guarantee.  
This README summarises both options, their trade-offs, and concrete recommendations.

---

## 1  Quick Comparison

| Metric | **Exact Gaussian KL** (`kl_mean` code) | **Schulman Surrogate** (`approx_kl` code) |
|--------|----------------------------------------|-------------------------------------------|
| Formula | Closed-form forward KL   | 2-nd-order Taylor expansion of same KL |
| Bias    | **None**                 | **Yes** – grows with policy drift |
| Variance| Low (analytic)           | Very low |
| ∇ Correctness | Exact (path-wise term suffices) | Biased unless score-function term added |
| Trust-region | Precise            | Under-penalises large updates |
| Scope   | Requires analytic KL (Normal, Categorical…) | Works for any policy with log-probs |

---

## 2  Code Snippets

<details>
<summary>Exact diagonal-Gaussian KL</summary>

```python
kl_mean = torch.mean(
    torch.sum(
        torch.log(newsigma / old_sigma + 1e-5)
        + (old_sigma.pow(2) + (old_mu - new_mu).pow(2))
          / (2 * newsigma.pow(2))
        - 0.5,
        dim=-1,
    )
)
```
</details>

<details>
<summary>Schulman Surrogate</summary>

```python
kl_mean = ((ratio - 1) - logratio).mean()
```
</details>

## 3  Impact on Training

### 3.1  Exact KL
	•	Preserves the intended KL budget → smooth learning curves.
	•	Slightly higher per-step variance (still tiny; analytic).
	•	Gradient is exact without extra REINFORCE terms.

### 3.2  Schulman Surrogate
	•	Great for tiny updates (very low variance).
	•	Under-estimates real KL once drift exceeds the trust-region radius.
	•	Autograd gives a biased ∇KL unless you add logratio * grad_logprob (the score-function term).
	•	Leads to KL spikes, oscillations, or premature entropy collapse in practice.
