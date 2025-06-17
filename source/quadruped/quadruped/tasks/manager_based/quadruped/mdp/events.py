import torch
from isaaclab.envs import ManagerBasedRLEnv


def random_termination(
    env: ManagerBasedRLEnv, probability: float = 0.01
) -> torch.Tensor:
    """Randomly terminate episodes based on a given probability.

    This function introduces stochasticity into episode termination, which can help:
    - Improve policy robustness by creating diverse episode lengths
    - Encourage exploration by preventing overly long episodes
    - Simulate unexpected events or failures in real-world scenarios

    Usage examples:
    - For training robustness: probability = 0.002 (0.2% per step)
    - For aggressive exploration: probability = 0.01 (1% per step)
    - For rare random events: probability = 0.0001 (0.01% per step)

    Args:
        env: The environment instance.
        probability: The probability of termination per step (default: 0.01).
                    Should be a small value (e.g., 0.001-0.01) to avoid
                    terminating episodes too frequently.

    Returns:
        Boolean tensor indicating which environments should be terminated.
    """
    # Generate random values for each environment
    random_values = torch.rand(env.num_envs, device=env.device)
    # Return True where random value is less than probability
    return random_values < probability
