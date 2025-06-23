# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
import time
from typing import List, Optional

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaaclab.utils import configclass
from isaaclab.utils.dict import print_dict
from models import CNNPPOAgent, MLPPPOAgent
from utils import seed_everything
# Import reward configuration for modification
from quadruped.tasks.manager_based.quadruped.quadruped_env_cfg import (
    SpotVelocityRewardsCfg,
)


@configclass
class EnvConfig:
    """Environment configuration"""

    task: str = "Spot-Velocity-Flat-Obstacle-Quadruped-v0"
    env_cfg_entry_point: str = "env_cfg_entry_point"
    num_envs: int = 4096
    seed: int = 1
    capture_video: bool = True
    video: bool = False
    video_length: int = 200
    video_interval: int = 2000
    disable_fabric: bool = False
    distributed: bool = False
    headless: bool = False
    enable_cameras: bool = False


@configclass
class ExperimentConfig:
    """PPO algorithm configuration"""

    exp_name: str = "ppo_continuous_action"
    torch_deterministic: bool = True
    device: str = "cuda:0"
    total_timesteps: int = 10_000_000
    learning_rate: float = 1e-3
    num_steps: int = 24
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 5
    norm_adv: bool = False
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.005
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.01
    init_at_random_ep_len: bool = False
    measure_burnin: int = 3
    checkpoint_interval: int = 10
    num_eval_envs: int = 3
    num_eval_env_steps: int = 200
    log_interval: int = 10


@configclass
class AgentConfig:
    """Neural network agent configuration"""

    agent_type: str = "MLPPPOAgent"
    actor_hidden_dims: List[int] = None
    critic_hidden_dims: List[int] = None
    actor_activation: str = "ELU"  # Will be converted to nn.ELU
    noise_std_type: str = "scalar"
    init_noise_std: float = 1.0

    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [512, 256, 128]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [512, 256, 128]


@configclass
class RewardConfig:
    """Reward function configuration"""

    # Positive rewards
    base_linear_velocity_weight: float = 20.0
    base_angular_velocity_weight: float = 20.0
    air_time_weight: float = 10.0
    foot_clearance_weight: float = 0.1
    gait_weight: float = 2.5
    survival_bonus_weight: float = 1.0e-2

    # Negative rewards (penalties)
    joint_torques_weight: float = -5.0e-4
    joint_acc_weight: float = -1.0e-4
    action_smoothness_weight: float = -1.0e-2
    base_motion_weight: float = -2.0
    base_orientation_weight: float = -1.0

    # Disabled rewards (set to 0.0)
    air_time_variance_weight: float = 0.0
    foot_slip_weight: float = 0.0
    joint_pos_weight: float = 0.0
    joint_vel_weight: float = 0.0


@configclass
class WandbConfig:
    """Weights & Biases configuration"""

    project: str = "ppo_continuous_action"
    log: bool = True
    log_video: bool = False
    tags: Optional[List[str]] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["ppo", "quadruped"]


@configclass
class Config:
    """Main configuration class"""

    env: EnvConfig = EnvConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    agent: AgentConfig = AgentConfig()
    rewards: RewardConfig = RewardConfig()
    wandb: WandbConfig = WandbConfig()

    # Computed at runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def create_custom_reward_cfg(reward_config: RewardConfig):
    """Create a custom reward configuration class based on the config"""

    class CustomSpotVelocityRewardsCfg(SpotVelocityRewardsCfg):
        def __post_init__(self):
            super().__post_init__()

            # Update reward weights from config
            self.base_linear_velocity.weight = reward_config.base_linear_velocity_weight
            self.base_angular_velocity.weight = (
                reward_config.base_angular_velocity_weight
            )
            self.air_time.weight = reward_config.air_time_weight
            self.foot_clearance.weight = reward_config.foot_clearance_weight
            self.gait.weight = reward_config.gait_weight
            self.joint_torques.weight = reward_config.joint_torques_weight
            self.joint_acc.weight = reward_config.joint_acc_weight
            self.action_smoothness.weight = reward_config.action_smoothness_weight
            self.base_motion.weight = reward_config.base_motion_weight
            self.base_orientation.weight = reward_config.base_orientation_weight
            self.air_time_variance.weight = reward_config.air_time_variance_weight
            self.foot_slip.weight = reward_config.foot_slip_weight
            self.joint_pos.weight = reward_config.joint_pos_weight
            self.joint_vel.weight = reward_config.joint_vel_weight

            # Update survival bonus
            if hasattr(self, "survival_bonus"):
                self.survival_bonus.weight = reward_config.survival_bonus_weight

    return CustomSpotVelocityRewardsCfg


def launch_app(cfg):
    """Launch Isaac Lab application"""
    from argparse import Namespace

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(Namespace(**OmegaConf.to_container(cfg, resolve=True)))
    return app_launcher.app


def make_isaaclab_env(
    task,
    device,
    num_envs,
    capture_video,
    disable_fabric,
    reward_cfg_class=None,
    log_dir=None,
    video_length=200,
    max_total_steps=None,
    *args,
    **kwargs,
):
    """Create Isaac Lab environment with custom reward configuration"""
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.torchrl import (
        IsaacLabRecordEpisodeStatistics,
        IsaacLabVecEnvWrapper,
    )
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    import quadruped.tasks  # noqa: F401

    def thunk():
        cfg = parse_env_cfg(
            task, device, num_envs=num_envs, use_fabric=not disable_fabric
        )

        # Override reward configuration if provided
        if reward_cfg_class is not None:
            cfg.rewards = reward_cfg_class()

        env = gym.make(
            task,
            cfg=cfg,
            render_mode="rgb_array"
            if (capture_video and log_dir is not None)
            else None,
            max_total_steps=max_total_steps,
        )
        print_dict({"max_episode_steps": env.unwrapped.max_episode_length}, nesting=4)
        env = IsaacLabRecordEpisodeStatistics(env)
        env = IsaacLabVecEnvWrapper(env, clip_actions=1.0)

        if capture_video and log_dir is not None:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": video_length,
                "disable_logger": True,
            }
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        return env

    return thunk


def get_activation_fn(activation_name: str):
    """Convert activation name to PyTorch activation function"""
    activation_map = {
        "ReLU": nn.ReLU,
        "ELU": nn.ELU,
        "Tanh": nn.Tanh,
        "GELU": nn.GELU,
        "LeakyReLU": nn.LeakyReLU,
    }
    return activation_map.get(activation_name, nn.ELU)


def setup_wandb(cfg: DictConfig, output_dir: str):
    """Setup wandb with Hydra configuration and sync directories"""
    # Convert hydra config to flat dict for wandb
    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    # Create run name that includes key hyperparameters
    run_name = f"{cfg.env.task}__{cfg.experiment.exp_name}__{cfg.env.seed}"
    if hasattr(cfg.experiment, "learning_rate"):
        run_name += f"__lr{cfg.experiment.learning_rate}"
    if hasattr(cfg.experiment, "ent_coef"):
        run_name += f"__ent{cfg.experiment.ent_coef}"

    # Get hydra job information for grouping
    job_name = "single_run"
    job_id = None
    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        job_name = hydra_cfg.job.name if hydra_cfg.job.name else "single_run"
        job_id = hydra_cfg.job.num if hasattr(hydra_cfg.job, "num") else None
    except:
        pass

    # Initialize wandb in the same directory as Hydra outputs
    wandb_kwargs = {
        "project": cfg.wandb.project,
        "name": run_name,
        "config": wandb_config,
        "save_code": True,
        "group": job_name,
        "dir": output_dir,  # Sync with Hydra output directory
    }

    # Add job id for multirun identification
    if job_id is not None:
        wandb_kwargs["job_type"] = f"job_{job_id}"

    # Add tags if specified
    if cfg.wandb.tags:
        wandb_kwargs["tags"] = cfg.wandb.tags

    # Add notes if specified
    if cfg.wandb.notes:
        wandb_kwargs["notes"] = cfg.wandb.notes

    # Handle dry run mode
    if not cfg.wandb.log:
        os.environ["WANDB_MODE"] = "dryrun"

    return wandb.init(**wandb_kwargs)


def train_ppo(cfg: DictConfig):
    """Main PPO training function"""
    # Calculate derived values
    cfg.batch_size = int(cfg.env.num_envs * cfg.experiment.num_steps)
    cfg.minibatch_size = int(cfg.batch_size // cfg.experiment.num_minibatches)
    cfg.num_iterations = cfg.experiment.total_timesteps // cfg.batch_size

    # Get current working directory (Hydra output directory)
    output_dir = os.getcwd()

    # Initialize Isaac Lab
    try:
        simulation_app = launch_app(cfg)
    except ImportError:
        raise ImportError("Isaac Lab is not installed. Please install it first.")

    try:
        # Setup wandb with hydra config
        run = setup_wandb(cfg, output_dir)

        # Create custom reward configuration
        custom_reward_cfg = create_custom_reward_cfg(cfg.rewards)

        # Prepare checkpoint directory
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # tracking best model
        best_return = -float("inf")
        best_step = 0
        best_ckpt = None

        device = (
            torch.device(cfg.experiment.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # env setup
        envs = make_isaaclab_env(
            cfg.env.task,
            cfg.experiment.device,
            cfg.env.num_envs,
            cfg.env.capture_video,
            cfg.env.disable_fabric,
            reward_cfg_class=custom_reward_cfg,
            max_total_steps=cfg.experiment.total_timesteps,
        )()

        # TRY NOT TO MODIFY: seeding
        seed_everything(
            envs,
            cfg.env.seed,
            use_torch=True,
            torch_deterministic=True,
        )

        n_obs = int(np.prod(envs.observation_space["policy"].shape[1:]))
        n_act = int(np.prod(envs.action_space.shape[1:]))
        assert isinstance(envs.action_space, gym.spaces.Box), (
            "only continuous action space is supported"
        )

        # Create agent with custom configuration
        activation_fn = get_activation_fn(cfg.agent.actor_activation)

        if cfg.agent.agent_type == "CNNPPOAgent":
            agent = CNNPPOAgent(n_obs, n_act).to(device)
        else:
            agent = MLPPPOAgent(
                n_obs,
                n_act,
                actor_hidden_dims=cfg.agent.actor_hidden_dims,
                critic_hidden_dims=cfg.agent.critic_hidden_dims,
                actor_activation=activation_fn,
                noise_std_type=cfg.agent.noise_std_type,
                init_noise_std=cfg.agent.init_noise_std,
            ).to(device)

        optimizer = optim.Adam(
            agent.parameters(), lr=cfg.experiment.learning_rate, eps=1e-5
        )

        # ALGO Logic: Storage setup
        obs = torch.zeros(
            (cfg.experiment.num_steps,) + envs.observation_space["policy"].shape
        ).to(device)
        actions = torch.zeros((cfg.experiment.num_steps,) + envs.action_space.shape).to(
            device
        )
        logprobs = torch.zeros((cfg.experiment.num_steps, cfg.env.num_envs)).to(device)
        rewards = torch.zeros((cfg.experiment.num_steps, cfg.env.num_envs)).to(device)
        dones = torch.zeros((cfg.experiment.num_steps, cfg.env.num_envs)).to(device)
        values = torch.zeros((cfg.experiment.num_steps, cfg.env.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        run_name = f"{cfg.env.task}__{cfg.experiment.exp_name}__{cfg.env.seed}"

        next_obs, _ = envs.reset()
        next_done = torch.zeros(cfg.env.num_envs).to(device)

        # randomize initial episode lengths (for exploration)
        if cfg.experiment.init_at_random_ep_len:
            envs.unwrapped.episode_length_buf = torch.randint_like(
                envs.unwrapped.episode_length_buf,
                high=int(envs.unwrapped.max_episode_length),
            )

        max_ep_ret = -float("inf")
        max_ep_reward = -float("inf")
        (
            success_rates,
            avg_reward_per_step,
            avg_returns,
            max_ep_length,
            goals_reached,
        ) = ([], [], [], [], [])

        # Create progress bars
        iteration_pbar = tqdm.tqdm(
            total=cfg.num_iterations, desc="Iterations", position=0, leave=True
        )
        step_pbar = tqdm.tqdm(
            total=cfg.experiment.num_steps, desc="Steps", position=1, leave=True
        )

        global_step_burnin = None
        start_time = None

        if cfg.wandb.log_video:
            video_frames = []
            indices = torch.randperm(cfg.env.num_envs)[: min(9, cfg.env.num_envs)].to(
                device
            )

        for iteration in range(1, cfg.num_iterations + 1):
            if iteration == cfg.experiment.measure_burnin:
                global_step_burnin = global_step
                start_time = time.time()

            # Annealing the rate if instructed to do so.
            if cfg.experiment.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
                lrnow = frac * cfg.experiment.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            step_pbar.reset()

            for step in range(0, cfg.experiment.num_steps):
                global_step += cfg.env.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, next_done, infos = envs.step(action)
                if "episode" in infos:
                    for r in infos["episode"]["r"]:
                        max_ep_ret = max(max_ep_ret, r)
                        avg_returns.append(r)
                    for r in infos["episode"]["reward_max"]:
                        max_ep_reward = max(max_ep_reward, r)
                        avg_reward_per_step.append(r)
                if "success_rate" in infos:
                    success_rates.append(infos["success_rate"])
                if "max_episode_length" in infos:
                    max_ep_length.append(infos["max_episode_length"])
                if "goals_reached" in infos:
                    goals_reached.append(infos["goals_reached"])
                if cfg.wandb.log_video:
                    frame = next_obs[indices, : 3 * 32 * 32].reshape(-1, 3, 32, 32)
                    frame = (
                        torchvision.utils.make_grid(frame, nrow=3, scale_each=True)
                        * 255.0
                    )
                    video_frames.append(frame)
                rewards[step] = reward.view(-1)
                step_pbar.update(1)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(cfg.experiment.num_steps)):
                    if t == cfg.experiment.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + cfg.experiment.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + cfg.experiment.gamma
                        * cfg.experiment.gae_lambda
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.observation_space["policy"].shape[1:])
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.action_space.shape[1:])
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(cfg.batch_size)
            clipfracs = []
            for epoch in range(cfg.experiment.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, cfg.batch_size, cfg.minibatch_size):
                    end = start + cfg.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > cfg.experiment.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if cfg.experiment.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1 - cfg.experiment.clip_coef,
                        1 + cfg.experiment.clip_coef,
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if cfg.experiment.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -cfg.experiment.clip_coef,
                            cfg.experiment.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = v_loss_max.mean()
                    else:
                        v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - cfg.experiment.ent_coef * entropy_loss
                        + v_loss * cfg.experiment.vf_coef
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        agent.parameters(), cfg.experiment.max_grad_norm
                    )
                    optimizer.step()

                if (
                    cfg.experiment.target_kl is not None
                    and approx_kl > cfg.experiment.target_kl
                ):
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            if (
                global_step_burnin is not None
                and iteration % cfg.experiment.log_interval == 0
            ):
                speed = (global_step - global_step_burnin) / (time.time() - start_time)
                desc = f"gl_st={global_step:3.0F}, "

                with torch.no_grad():
                    logs = {
                        "charts/learning_rate": optimizer.param_groups[0]["lr"],
                        "losses/value_loss": v_loss.item(),
                        "losses/policy_loss": pg_loss.item(),
                        "losses/entropy": entropy_loss.item(),
                        "losses/old_approx_kl": old_approx_kl.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "losses/clipfrac": np.mean(clipfracs),
                        "losses/explained_variance": explained_var,
                        "charts/SPS": speed,
                    }

                    if len(avg_returns) > 0:
                        logs["charts/episodic_return"] = torch.tensor(
                            avg_returns
                        ).mean()
                        logs["charts/max_episodic_return"] = max_ep_ret
                        desc += f"ep_ret : avg={torch.tensor(avg_returns).mean():.2f}, max={max_ep_ret:.2f}, "
                    if len(avg_reward_per_step) > 0:
                        logs["charts/episodic_length"] = torch.tensor(
                            avg_reward_per_step
                        ).mean()
                        desc += f"rew_per_step : avg={torch.tensor(avg_reward_per_step).mean():.2f}, max={max_ep_reward:.2f}"
                    if len(success_rates) > 0:
                        logs["charts/success_rate"] = torch.tensor(success_rates).mean()
                    if len(max_ep_length) > 0:
                        logs["charts/max_episode_length"] = torch.tensor(
                            max_ep_length
                        ).mean()
                    if len(goals_reached) > 0:
                        logs["charts/goals_reached"] = torch.tensor(
                            goals_reached
                        ).mean()

                    # Reset tracking lists
                    (
                        success_rates,
                        avg_reward_per_step,
                        avg_returns,
                        max_ep_length,
                        goals_reached,
                    ) = ([], [], [], [], [])

                iteration_pbar.set_description(f"spd(sps): {speed:3.1f}, " + desc)

                # Log to wandb
                wandb.log(logs, step=global_step)

                if cfg.wandb.log_video and len(video_frames) > 0:
                    video_tensor = torch.stack(video_frames)
                    wandb.log(
                        {
                            "videos/obs_grid": wandb.Video(
                                video_tensor.detach().cpu().numpy().astype(np.uint8),
                                fps=25,
                                format="mp4",
                            )
                        },
                        step=global_step,
                    )
                    video_frames = []
                    indices = torch.randperm(cfg.env.num_envs)[
                        : min(9, cfg.env.num_envs)
                    ].to(device)

            # save every checkpoint_interval steps
            if (
                global_step_burnin is not None
                and iteration % cfg.experiment.checkpoint_interval == 0
            ):
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_{global_step}.pt")
                torch.save(
                    {
                        "model_state_dict": agent.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "global_step": global_step,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    ckpt_path,
                )

                if len(avg_returns) > 0:
                    mean_return = float(torch.tensor(avg_returns).mean().nan_to_num())
                    mean_return += 1e-9

                    if mean_return > best_return:
                        best_return = mean_return
                        best_step = global_step
                        best_ckpt = ckpt_path

            iteration_pbar.update(1)

        # Close progress bars
        step_pbar.close()
        iteration_pbar.close()

        envs.close()
        del envs

        # upload the best checkpoint as a wandb Artifact
        if best_ckpt is not None and cfg.wandb.log:
            artifact = wandb.Artifact(
                name="ppo-best-checkpoint",
                type="model",
                description=f"Best PPO model at step {best_step} with return {best_return:.2f}",
            )
            artifact.add_file(best_ckpt)
            run.log_artifact(artifact)

        # Log final summary
        wandb.summary["best_return"] = best_return
        wandb.summary["best_step"] = best_step
        wandb.summary["total_steps"] = global_step

        wandb.finish()

    except Exception as e:
        print("Exception:", e)
        raise
    finally:
        simulation_app.close()


# Register configuration with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(
    version_base=None, config_path="configs", config_name="ppo_continuous_action"
)
def main(cfg: DictConfig) -> None:
    train_ppo(cfg)


def main_no_hydra():
    """Main function for running without Hydra (backward compatibility)"""
    # Load YAML config and convert to DictConfig
    import yaml

    config_path = os.path.join(
        os.path.dirname(__file__), "configs", "torch", "ppo_continuous_action.yaml"
    )

    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    # Convert to structured config
    cfg = OmegaConf.create(yaml_config)

    # Set up basic structure if not present
    if "env" not in cfg:
        cfg.env = {}
    if "experiment" not in cfg:
        cfg.experiment = {}
    if "agent" not in cfg:
        cfg.agent = {}
    if "rewards" not in cfg:
        cfg.rewards = {}
    if "wandb" not in cfg:
        cfg.wandb = {}

    # Map flat config to structured config
    env_fields = [
        "task",
        "num_envs",
        "seed",
        "capture_video",
        "video",
        "video_length",
        "video_interval",
        "disable_fabric",
        "distributed",
        "headless",
        "enable_cameras",
    ]
    exp_fields = [
        "exp_name",
        "torch_deterministic",
        "device",
        "total_timesteps",
        "learning_rate",
        "num_steps",
        "anneal_lr",
        "gamma",
        "gae_lambda",
        "num_minibatches",
        "update_epochs",
        "norm_adv",
        "clip_coef",
        "clip_vloss",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
        "init_at_random_ep_len",
        "measure_burnin",
        "checkpoint_interval",
        "num_eval_envs",
        "num_eval_env_steps",
        "log_interval",
    ]

    for field in env_fields:
        if field in yaml_config:
            cfg.env[field] = yaml_config[field]

    for field in exp_fields:
        if field in yaml_config:
            cfg.experiment[field] = yaml_config[field]

    # Set defaults for missing fields
    cfg.agent.agent_type = yaml_config.get("agent_type", "MLPPPOAgent")
    cfg.wandb.log = yaml_config.get("log", False)
    cfg.wandb.log_video = yaml_config.get("log_video", False)

    # Set default reward weights
    rewards_cfg = RewardConfig()
    for field in rewards_cfg.__dict__:
        if field not in cfg.rewards:
            cfg.rewards[field] = getattr(rewards_cfg, field)

    train_ppo(cfg)


if __name__ == "__main__":
    # Check if running with Hydra or not
    if (
        "--config-path" in sys.argv
        or "--config-name" in sys.argv
        or "-cp" in sys.argv
        or "-cn" in sys.argv
    ):
        main()
    else:
        # Try to run with Hydra by default
        try:
            main()
        except:
            # Fallback to non-Hydra mode
            print("Running in non-Hydra mode (backward compatibility)")
            main_no_hydra()
