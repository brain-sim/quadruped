import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from scripts.models import AGENT_LOOKUP_BY_ALGORITHM
from scripts.utils import EmpiricalNormalization, adjust_noise_scales


class TorchScriptNormalizer(nn.Module):
    """
    TorchScript-compatible normalizer that replicates EmpiricalNormalization
    functionality without the problematic shape check.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply normalization without shape checking
        return (x - self._mean) / (self._std + self.eps)


class ModelWithNormalizer(nn.Module):
    """
    Wrapper class that combines a model with an observation normalizer.
    This allows both to be traced together as a single TorchScript module.
    """

    def __init__(
        self,
        model: nn.Module,
        normalizer: Optional[nn.Module] = None,
        action_bounds: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.normalizer = normalizer if normalizer is not None else nn.Identity()
        self.register_buffer("action_bounds", torch.tensor(action_bounds))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Apply normalization if available
        normalized_obs = self.normalizer(obs)
        # Forward through the model
        return self.model(normalized_obs).clamp(-1, 1) * self.action_bounds


def convert_checkpoint_to_jit(
    checkpoint_path: str,
    output_path: str,
    algorithm: str = "fast_td3",
    obs_type: str = "state",
    num_eval_envs: int = 1,
    device: str = "cpu",
    **model_kwargs,
) -> torch.jit.ScriptModule:
    """
    Convert a trained model checkpoint to TorchScript format.

    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Path to save the TorchScript model
        algorithm: Algorithm type ('ppo', 'td3', 'fast_td3')
        obs_type: Observation type ('state', 'image')
        num_eval_envs: Number of environments for evaluation (affects noise scales)
        device: Device to load the model on
        **model_kwargs: Additional model parameters

    Returns:
        Traced TorchScript model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model parameters from checkpoint args
    args = checkpoint.get("args", {})
    n_obs = args.get("observation_space", 48)  # Default for Spot
    n_act = args.get("action_space", 12)  # Default for Spot

    # Get model classes based on algorithm and observation type
    agent_classes = AGENT_LOOKUP_BY_ALGORITHM[algorithm][obs_type]

    if isinstance(agent_classes, (list, tuple)):
        # For TD3/FastTD3: [Actor, Critic]
        actor_cls, critic_cls = agent_classes
        print(
            f"Loading {algorithm} model with {actor_cls.__name__} and {critic_cls.__name__}"
        )

        # Initialize actor with required parameters
        actor_params = {
            "n_obs": n_obs,
            "n_act": n_act,
            "num_envs": num_eval_envs,
            "device": torch.device(device),
            **model_kwargs,
        }

        # Add algorithm-specific parameters
        if algorithm == "fast_td3":
            actor_params.update(
                {
                    "init_scale": args.get("init_scale", 0.01),
                    "std_min": args.get("std_min", 0.001),
                    "std_max": args.get("std_max", 0.05),
                    "hidden_dims": args.get("actor_hidden_dims", [512, 256, 128]),
                }
            )
        elif algorithm == "td3":
            actor_params.update(
                {
                    "a_max": args.get("action_bounds", 1.0),
                    "a_min": -args.get("action_bounds", 1.0),
                    "exploration_noise": args.get("exploration_noise", 0.1),
                    "hidden_dims": args.get("actor_hidden_dims", [512, 256, 128]),
                }
            )

        model = actor_cls(**actor_params)

    else:
        # For PPO: single agent class
        model = agent_classes(n_obs=n_obs, n_act=n_act, **model_kwargs)
        print(f"Loading {algorithm} model with {agent_classes.__name__}")

    # Load state dict
    if "actor_state_dict" in checkpoint:
        state_dict = checkpoint["actor_state_dict"]
        # Adjust noise scales if needed for FastTD3
        if algorithm == "fast_td3":
            state_dict = adjust_noise_scales(state_dict, model, num_eval_envs)
        model.load_state_dict(state_dict)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Assume checkpoint is the state_dict

    model.eval()

    # Handle observation normalizer - always include it
    normalizer = nn.Identity()  # Default to Identity
    if "obs_normalizer_state" in checkpoint:
        obs_normalizer_state = checkpoint["obs_normalizer_state"]
        if obs_normalizer_state is not None:
            # Extract mean and std from the normalizer state
            mean = obs_normalizer_state["_mean"].squeeze(0)  # Remove batch dimension
            std = obs_normalizer_state["_std"].squeeze(0)  # Remove batch dimension
            eps = args.get("obs_normalization_eps", 1e-2)

            # Create TorchScript-compatible normalizer
            normalizer = TorchScriptNormalizer(mean, std, eps)
            print("Loaded observation normalizer from checkpoint")

    # Create wrapper model that includes normalizer
    wrapper_model = ModelWithNormalizer(
        model,
        normalizer,
        action_bounds=checkpoint["args"].get("action_bounds", 1.0),
    )
    wrapper_model.eval()

    # Create example input for tracing
    example_input = torch.randn(1, n_obs, device=device)

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper_model, example_input)

    # Save the traced model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced_model.save(output_path)
    print(f"JIT model saved to {output_path}")

    return traced_model


def test_non_traced_model(
    test_input: torch.Tensor,
    checkpoint_path: str,
    algorithm: str = "fast_td3",
    obs_type: str = "state",
    num_eval_envs: int = 1,
    device: str = "cpu",
    **model_kwargs,
) -> torch.Tensor:
    """
    Test the non-traced model with sample input.

    Args:
        checkpoint_path: Path to the checkpoint file
        algorithm: Algorithm type
        obs_type: Observation type
        num_eval_envs: Number of environments
        device: Device to run the test on
        **model_kwargs: Additional model parameters

    Returns:
        Output from the non-traced model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model parameters from checkpoint args
    args = checkpoint.get("args", {})
    n_obs = args.get("observation_space", 48)
    n_act = args.get("action_space", 12)

    # Get model classes based on algorithm and observation type
    agent_classes = AGENT_LOOKUP_BY_ALGORITHM[algorithm][obs_type]

    if isinstance(agent_classes, (list, tuple)):
        actor_cls, critic_cls = agent_classes
        actor_params = {
            "n_obs": n_obs,
            "n_act": n_act,
            "num_envs": num_eval_envs,
            "device": torch.device(device),
            **model_kwargs,
        }

        if algorithm == "fast_td3":
            actor_params.update(
                {
                    "init_scale": args.get("init_scale", 0.01),
                    "std_min": args.get("std_min", 0.001),
                    "std_max": args.get("std_max", 0.05),
                    "hidden_dims": args.get("actor_hidden_dims", [512, 256, 128]),
                }
            )
        elif algorithm == "td3":
            actor_params.update(
                {
                    "a_max": args.get("action_bounds", 1.0),
                    "a_min": -args.get("action_bounds", 1.0),
                    "exploration_noise": args.get("exploration_noise", 0.1),
                    "hidden_dims": args.get("actor_hidden_dims", [512, 256, 128]),
                }
            )

        model = actor_cls(**actor_params)
    else:
        model = agent_classes(n_obs=n_obs, n_act=n_act, **model_kwargs)

    # Load state dict
    if "actor_state_dict" in checkpoint:
        state_dict = checkpoint["actor_state_dict"]
        if algorithm == "fast_td3":
            state_dict = adjust_noise_scales(state_dict, model, num_eval_envs)
        model.load_state_dict(state_dict)
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Handle observation normalizer
    normalizer = nn.Identity()
    if "obs_normalizer_state" in checkpoint:
        obs_normalizer_state = checkpoint["obs_normalizer_state"]
        if obs_normalizer_state is not None:
            # Use the original EmpiricalNormalization for comparison
            normalizer = EmpiricalNormalization(
                shape=n_obs,
                device=torch.device(device),
            )
            normalizer.load_state_dict(obs_normalizer_state)
            normalizer.eval()

    # Create wrapper model
    wrapper_model = ModelWithNormalizer(
        model,
        normalizer,
        action_bounds=checkpoint["args"].get("action_bounds", 1.0),
    )
    wrapper_model.eval()

    # Run inference
    with torch.no_grad():
        output = wrapper_model(test_input)

    return output


def test_traced_model(
    test_input: torch.Tensor, model_path: str, device: str = "cpu"
) -> torch.Tensor:
    """
    Test the traced model with sample input.

    Args:
        test_input: Test input
        model_path: Path to the traced model
        device: Device to run the test on

    Returns:
        Output from the traced model
    """
    # Load the traced model
    traced_model = torch.jit.load(model_path, map_location=device)

    # Run inference
    with torch.no_grad():
        output = traced_model(test_input)

    return output


def compare_models(
    checkpoint_path: str,
    traced_model_path: str,
    algorithm: str = "fast_td3",
    obs_type: str = "state",
    num_eval_envs: int = 1,
    n_obs: int = 48,
    device: str = "cpu",
    **model_kwargs,
) -> None:
    """
    Compare outputs from non-traced and traced models.

    Args:
        checkpoint_path: Path to the checkpoint file
        traced_model_path: Path to the traced model
        algorithm: Algorithm type
        obs_type: Observation type
        num_eval_envs: Number of environments
        device: Device to run the test on
        **model_kwargs: Additional model parameters
    """
    print("Testing non-traced model...")
    test_input = torch.zeros(1, n_obs, device=device)
    test_input[:, 8] = -1.0
    non_traced_output = test_non_traced_model(
        test_input=test_input,
        checkpoint_path=checkpoint_path,
        algorithm=algorithm,
        obs_type=obs_type,
        num_eval_envs=num_eval_envs,
        device=device,
        **model_kwargs,
    )

    print("Testing traced model...")
    traced_output = test_traced_model(
        test_input=test_input,
        model_path=traced_model_path,
        device=device,
    )

    print("\n" + "=" * 50)
    print("COMPARISON RESULTS")
    print("=" * 50)
    print(f"Non-traced output shape: {non_traced_output.shape}")
    print(f"Traced output shape: {traced_output.shape}")
    print(f"Non-traced output: {non_traced_output}")
    print(f"Traced output: {traced_output}")
    print(
        f"Non-traced output range: [{non_traced_output.min().item():.6f}, {non_traced_output.max().item():.6f}]"
    )
    print(
        f"Traced output range: [{traced_output.min().item():.6f}, {traced_output.max().item():.6f}]"
    )

    # Calculate differences
    diff = torch.abs(non_traced_output - traced_output)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Maximum absolute difference: {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")

    if max_diff < 1e-6:
        print("✅ Models produce identical outputs (within numerical precision)")
    elif max_diff < 1e-4:
        print("✅ Models produce very similar outputs (minor numerical differences)")
    else:
        print("❌ Models produce significantly different outputs")

    print("=" * 50)


if __name__ == "__main__":
    # Example usage for FastTD3
    checkpoint_path = "/home/chandramouli/quadruped/wandb/run-20250710_124753-6z8oy4u1/files/checkpoints/ckpt_138000.pt"
    output_path = "/home/chandramouli/cognitiverl/source/cognitiverl/cognitiverl/tasks/direct/custom_assets/spot_policy_test_v2.pt"

    # Convert checkpoint to TorchScript
    traced_model = convert_checkpoint_to_jit(
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        algorithm="fast_td3",
        obs_type="state",
        num_eval_envs=1,
        device="cpu",
    )

    # Compare the models
    compare_models(
        checkpoint_path=checkpoint_path,
        traced_model_path=output_path,
        algorithm="fast_td3",
        obs_type="state",
        num_eval_envs=1,
        device="cpu",
    )
