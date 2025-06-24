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