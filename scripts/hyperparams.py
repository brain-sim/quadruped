import argparse
import os
from dataclasses import asdict, dataclass, fields

import yaml


@dataclass
class EnvArgs:
    task: str = "Spot-Velocity-Flat-v0"
    """the id of the environment"""
    env_cfg_entry_point: str = "env_cfg_entry_point"
    """the entry point of the environment configuration"""
    num_envs: int = 4096
    """the number of parallel environments to simulate"""
    seed: int = 1
    """seed of the environment"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    video: bool = False
    """record videos during training"""
    video_length: int = 200
    """length of the recorded video (in steps)"""
    video_interval: int = 2000
    """interval between video recordings (in steps)"""
    disable_fabric: bool = False
    """disable fabric and use USD I/O operations"""
    distributed: bool = False
    """run training with multiple GPUs or nodes"""
    headless: bool = False
    """run training in headless mode"""
    enable_cameras: bool = False
    """enable cameras to record sensor inputs."""


@dataclass
class PlayArg(EnvArgs):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda:0"
    """device to use for training"""
    checkpoint_path: str = "/home/chandramouli/quadruped/wandb/run-20250701_143749-04yfv2oo/files/checkpoints/ckpt_1179648000.pt"
    """path to the checkpoint to load"""
    num_envs: int = 32
    """number of environments to run for evaluation/play."""
    total_timesteps: int = 1000
    """number of steps to run for evaluation/play."""
    agent_type: str = "MLP"


@dataclass
class ExperimentArgs(EnvArgs):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda:0"
    """device to use for training"""
    """the id of the environment"""
    total_timesteps: int = 2_000_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.001
    """the learning rate of the optimizer"""
    num_steps: int = 24
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    init_at_random_ep_len: bool = False
    """randomize initial episode lengths (for exploration)"""
    measure_burnin: int = 3

    # Agent config
    agent_type: str = "MLP"

    checkpoint_interval: int = total_timesteps
    """environment steps between saving checkpoints."""
    log_interval: int = 10
    """number of iterations between logging."""
    log: bool = False
    """whether to log the training process."""
    log_video: bool = False
    """whether to log the video."""


@configclass
class PPOArgs(ExperimentArgs):
    total_timesteps: int = 2_000_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.001
    """the learning rate of the optimizer"""
    num_steps: int = 24
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 5
    """the K epochs to update the policy"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0025
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    # Adaptive learning rate parameters
    adaptive_lr: bool = True
    """Use adaptive learning rate based on KL divergence"""
    target_kl: float = 0.01
    """the target KL divergence threshold"""
    lr_multiplier: float = 1.5
    """Factor to multiply/divide learning rate by"""


@configclass
class SpotVelocityFlatPPOArgs(PPOArgs):
    task: str = "Spot-Velocity-Flat-v0"
    pass


@configclass
class SpotVelocityRoughPPOArgs(PPOArgs):
    task: str = "Spot-Velocity-Rough-v0"
    vf_coef: float = 1.0
    """coefficient of the value function"""
    ent_coef: float = 0.005
    """coefficient of the entropy"""


@configclass
class TD3Args(ExperimentArgs):
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 25_000
    """the replay memory buffer size"""


@configclass
class SpotVelocityFlatTD3Args(TD3Args):
    pass


@configclass
class SpotVelocityRoughTD3Args(TD3Args):
    pass


def _dataclass_to_argparse(parser, dataclass_type, prefix="", defaults=None):
    """
    Add arguments to parser from dataclass fields. Optionally use a prefix for nested dataclasses.
    """
    for f in fields(dataclass_type):
        arg_name = f"--{prefix}{f.name}"
        arg_type = f.type
        print(prefix, f.name, arg_name)
        default = getattr(defaults, f.name) if defaults is not None else f.default
        # Handle bools as store_true/store_false
        if arg_type is bool:
            if default is True:
                parser.add_argument(
                    arg_name, action="store_false", dest=f.name, help="(default: True)"
                )
            else:
                parser.add_argument(
                    arg_name, action="store_true", dest=f.name, help="(default: False)"
                )
        else:
            parser.add_argument(
                arg_name,
                type=arg_type,
                default=argparse.SUPPRESS,
                help=f"(default: {default})",
            )


def _update_dict(d, u):
    """Recursively update dict d with values from u."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_dict(d[k], v)
        else:
            d[k] = v
    return d


def load_args(ArgsClass, yaml_path=None, cli_args=None):
    """
    Loads arguments from dataclass defaults, then YAML, then CLI (in that order).
    Args:
        ArgsClass: The dataclass type to instantiate.
        yaml_path: Path to YAML file (optional, can be None).
        cli_args: List of CLI args (optional, defaults to sys.argv[1:]).
    Returns:
        An instance of ArgsClass with merged arguments.
    """
    # 1. Start with dataclass defaults
    default_args = ArgsClass()
    merged = asdict(default_args)

    # 2. If YAML, update
    if yaml_path is not None and os.path.isfile(yaml_path):
        with open(yaml_path, "r") as f:
            yaml_args = yaml.safe_load(f)
        if yaml_args:
            merged.update(yaml_args)

    # 3. CLI overrides
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file."
    )
    _dataclass_to_argparse(parser, ArgsClass, defaults=default_args)
    args_ns, unknown = parser.parse_known_args(cli_args)
    cli_dict = vars(args_ns)
    config_from_cli = cli_dict.pop("config", None)
    # If --config is given on CLI, reload YAML and update again
    if config_from_cli and (yaml_path is None or config_from_cli != yaml_path):
        if os.path.isfile(config_from_cli):
            with open(config_from_cli, "r") as f:
                yaml_args2 = yaml.safe_load(f)
            if yaml_args2:
                merged.update(yaml_args2)
    merged.update(cli_dict)
    # Return as ArgsClass instance
    return ArgsClass(**merged)
