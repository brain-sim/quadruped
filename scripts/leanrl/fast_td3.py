# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import os
import sys
import time
from collections import deque
from dataclasses import asdict
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from isaaclab.utils import configclass
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule
from torch.amp import GradScaler, autocast

import wandb
from scripts.buffers import SimpleReplayBuffer, TorchRLInfoLogger
from scripts.models import AGENT_LOOKUP_BY_ALGORITHM, AGENT_LOOKUP_BY_INPUT_TYPE
from scripts.utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    load_args,
    mark_step,
    print_dict,
    save_params,
    seed_everything,
)
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

# TODO: Batch size (global) and transition batch size should be different.
# The current code only works if they are both the same.


@configclass
class EnvArgs:
    task: str = "Spot-Velocity-Flat-v0"
    """the id of the environment"""
    env_cfg_entry_point: str = "env_cfg_entry_point"
    """the entry point of the environment configuration"""
    num_envs: int = 4096
    """the number of parallel environments to simulate"""
    seed: int = 1
    """seed of the environment"""
    capture_video: bool = False
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


@configclass
class ExperimentArgs:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    device: str = "cuda:0"
    """cuda:0 will be enabled by default"""

    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    actor_learning_rate: float = 3e-4
    """the learning rate of the actor"""
    critic_learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = 10240
    """the replay memory buffer size"""
    buffer_device: str = "cpu"
    """the device of the replay memory buffer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.1
    """target smoothing coefficient (default: 0.1)"""
    batch_size: int = 32768
    """the batch size of sample from the reply memory"""
    policy_noise: float = 0.001
    """the scale of policy noise"""
    learning_starts: int = 10
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    num_updates: int = 4
    """UTD ratio"""

    num_atoms: int = 251
    """the number of atoms for the distributional critic"""
    v_min: float = -50.0
    """the minimum value of the support"""
    v_max: float = 50.0
    """the maximum value of the support"""
    std_min: float = 0.001
    """the minimum value of the std"""
    std_max: float = 0.05
    """the maximum value of the std"""
    bootstrap: bool = True
    """whether to use bootstrap"""
    use_cdq: bool = True
    """whether to use Clipped Double Q-learning"""
    init_scale: float = 0.01
    """the initial scale of the action"""

    measure_burnin: int = 3

    # Agent config
    obs_type: str = "state"
    """the type of the observations"""
    algorithm: str = "fast_td3"
    """the algorithm to use"""
    obs_normalization: bool = True
    """whether to normalize the observations"""
    reward_normalization: bool = False
    """whether to normalize the rewards"""

    compile: bool = True
    """use torch.compile"""
    compile_mode: str = "reduce-overhead"
    """the mode of the torch.compile"""
    weight_decay: float = 0.1
    """the weight decay of the optimizer"""
    critic_learning_rate_end: float = 3e-4
    """the learning rate of the critic at the end of the training"""
    actor_learning_rate_end: float = 3e-4
    """the learning rate of the actor at the end of the training"""
    amp: bool = True
    """whether to use amp"""
    amp_dtype: str = "bf16"
    """the dtype of the amp"""
    num_steps: int = 8
    """number of steps to run the environment"""
    action_bounds: float = 1.0
    """the bounds of the action"""
    clip_actions: Optional[float] = None
    """whether to clip the actions"""
    log: bool = False
    """log to wandb"""
    log_interval: int = 100
    """log interval"""
    checkpoint_interval: int = 1000
    """checkpoint interval"""

    clip_grad_norm: bool = False
    """whether to clip the gradient norm"""
    max_grad_norm: float = 0.0
    """the maximum norm for the gradient clipping"""
    actor_hidden_dims: list[int] = [512, 256, 128]
    """the hidden dimensions of the actor"""
    critic_hidden_dims: list[int] = [1024, 512, 256]
    """the hidden dimensions of the critic"""

    def __post_init__(self):
        if hasattr(self, "num_envs"):
            self.buffer_device = self.device
        self.buffer_size = min(self.buffer_size, self.total_timesteps)


@configclass
class Args(ExperimentArgs, EnvArgs):
    pass


def launch_app(args):
    from argparse import Namespace

    app_launcher = AppLauncher(Namespace(**asdict(args)))
    return app_launcher.app


def get_args():
    return load_args(Args)


try:
    from isaaclab.app import AppLauncher

    args = get_args()
    simulation_app = launch_app(args)
except ImportError:
    raise ImportError("Isaac Lab is not installed. Please install it first.")


def make_env(task, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(task, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(task)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def make_isaaclab_env(
    task, seed, device, num_envs, capture_video, disable_fabric, **kwargs
):
    import isaaclab_tasks  # noqa: F401
    import quadruped.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    from scripts.wrappers import IsaacLabVecEnvWrapper

    def thunk():
        cfg = parse_env_cfg(
            task, device, num_envs=num_envs, use_fabric=not disable_fabric
        )
        cfg.seed = seed
        env = gym.make(
            task,
            cfg=cfg,
            render_mode="rgb_array" if capture_video else None,
        )
        env = IsaacLabVecEnvWrapper(
            env,
            action_bounds=kwargs.get("action_bounds", None),
            clip_actions=kwargs.get("clip_actions", None),
        )
        return env

    return thunk


def main(args):
    run_name = f"{args.exp_name}__{args.task}"
    amp_enabled = args.amp and "cuda" in args.device and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if "cuda" in args.device and torch.cuda.is_available()
        else "mps"
        if "mps" in args.device and torch.backends.mps.is_available()
        else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    run = wandb.init(
        project="fast_td3",
        name=f"{run_name}",
        config=vars(args),
        save_code=True,
    )

    # prepare local checkpoint directory inside the wandb run folder
    run_dir = run.dir
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )
    buffer_device = (
        torch.device(args.buffer_device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    # TRY NOT TO MODIFY: seeding
    seed_everything(
        args.seed, use_torch=True, torch_deterministic=args.torch_deterministic
    )
    # env setup
    envs = make_isaaclab_env(
        args.task,
        args.seed,
        args.device,
        args.num_envs,
        args.disable_fabric,
        args.capture_video,
        action_bounds=args.action_bounds,
        clip_actions=args.clip_actions,
    )()
    print_dict(args, color="green", attrs=["bold"])
    n_obs = int(np.prod(envs.num_obs))
    n_act = int(np.prod(envs.num_actions))
    action_low, action_high = -1.0, 1.0
    print_dict(
        f"action_low: {action_low}, action_high: {action_high}",
        color="yellow",
        attrs=["bold"],
    )
    assert isinstance(envs.action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
    else:
        obs_normalizer = nn.Identity()

    if args.reward_normalization:
        reward_normalizer = RewardNormalizer(
            gamma=args.gamma,
            device=device,
            g_max=min(abs(args.v_min), abs(args.v_max)),
        )
    else:
        reward_normalizer = nn.Identity()

    actor_class, critic_class = AGENT_LOOKUP_BY_ALGORITHM[args.algorithm][args.obs_type]
    actor = actor_class(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=args.num_envs,
        init_scale=args.init_scale,
        std_min=args.std_min,
        std_max=args.std_max,
        device=device,
        hidden_dims=args.actor_hidden_dims,
    )
    print(f"actor : {actor}")
    actor_detach = actor_class(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=args.num_envs,
        init_scale=args.init_scale,
        std_min=args.std_min,
        std_max=args.std_max,
        device=device,
        hidden_dims=args.actor_hidden_dims,
    )
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    qnet = critic_class(
        n_obs=n_obs,
        n_act=n_act,
        device=device,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dims=args.critic_hidden_dims,
    )
    print(f"qnet : {qnet}")

    # discard params of net
    qnet_target = critic_class(
        n_obs=n_obs,
        n_act=n_act,
        device=device,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dims=args.critic_hidden_dims,
    )
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )

    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )

    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_obs,
        asymmetric_obs=False,
        playground_mode=False,
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=buffer_device,
    )

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]

        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def update_main(data, log_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations, next_observations, actions, rewards, time_outs, dones = (
                data["observations"],
                data["next_observations"],
                data["actions"],
                data["rewards"],
                data["time_outs"],
                data["dones"],
            )

            # Ensure rewards and dones have correct dimensions for vmap
            dones = dones.bool()
            time_outs = time_outs.bool()
            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )
            if args.bootstrap:
                bootstrap = (time_outs | ~dones).float()
            else:
                bootstrap = (~dones).float()

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            if next_state_actions.isinf().any():
                raise ValueError("next_state_actions inf")
            discount = args.gamma ** data["effective_n_steps"]
            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        next_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )

                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)

                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist = qf1_next_target_projected
                    qf2_next_target_dist = qf2_next_target_projected

            qf1, qf2 = qnet(observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss
        # optimize the model
        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.clip_grad_norm:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)

        scaler.step(q_optimizer)
        scaler.update()

        log_dict["qf_loss"] = qf_loss.detach()
        log_dict["qf1_loss"] = qf1_loss.detach()
        log_dict["qf2_loss"] = qf2_loss.detach()
        log_dict["qf_max"] = qf1.max().detach()
        log_dict["qf_min"] = qf1.min().detach()
        log_dict["batch_rewards"] = rewards.mean().detach()
        log_dict["critic_grad_norm"] = critic_grad_norm.detach()
        return log_dict

    def update_pol(data, log_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            qf1, qf2 = qnet(observations, actor(observations))
            qf1_value = qnet.get_value(
                F.softmax(qf1, dim=-1)
            )  # TODO: Check if softmax is calculated on correct dimension
            qf2_value = qnet.get_value(
                F.softmax(qf2, dim=-1)
            )  # TODO: Check if softmax is calculated on correct dimension
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()
        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)

        if args.clip_grad_norm:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)

        scaler.step(actor_optimizer)
        scaler.update()
        log_dict["actor_loss"] = actor_loss.detach()
        log_dict["actor_grad_norm"] = actor_grad_norm.detach()
        return log_dict

    if args.compile:
        compile_mode = args.compile_mode
        update_main = torch.compile(update_main, mode=compile_mode)
        update_pol = torch.compile(update_pol, mode=compile_mode)
        policy = torch.compile(policy, mode=None)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    args.num_iterations = args.total_timesteps // args.num_envs
    pbar = tqdm.tqdm(range(args.num_iterations))
    start_time = None
    dones = None
    max_ep_ret = -float("inf")
    desc = ""
    log_transition = {}
    info_logger = TorchRLInfoLogger(device="cpu", buffer_size=1000)

    for global_step in pbar:
        mark_step()
        out_main = TensorDict()
        if global_step == args.measure_burnin:
            start_time = time.time()
            measure_burnin = global_step

        # ALGO LOGIC: put action logic here
        with (
            torch.no_grad(),
            autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled),
        ):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)
            actions = torch.clamp(actions, action_low, action_high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions.float())

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.clone()
        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
            terminations=infos["terminations"].long(),
            time_outs=infos["time_outs"].long(),
            dones=dones.long(),
            batch_size=(envs.num_envs,),
            device=buffer_device,
        )

        info_logger.update(
            infos,
            transition["observations"].max().item(),
            transition["observations"].min().item(),
            transition["actions"].max().item() * args.action_bounds,
            transition["actions"].min().item() * args.action_bounds,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step >= args.measure_burnin:
            speed = (
                (global_step - measure_burnin)
                * args.num_envs
                / (time.time() - start_time)
            )
        rb.extend(transition)
        if global_step > args.learning_starts:
            update_start_time = time.time()
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                data["observations"] = normalize_obs(data["observations"])
                data["next_observations"] = normalize_obs(data["next_observations"])
                data["rewards"] = normalize_reward(data["rewards"])
                out_main = update_main(data, out_main)
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        out_main = update_pol(data, out_main)
                else:
                    if global_step % args.policy_frequency == 0:
                        out_main = update_pol(data, out_main)

                # update the target networks
                # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
                soft_update(qnet, qnet_target, args.tau)

            if (
                args.log_interval > 0
                and global_step % args.log_interval == 0
                and start_time is not None
            ):
                update_time = time.time() - update_start_time
                desc = f"update time: {update_time: 4.4f}s"
                log_infos = info_logger.get_averaged_logs()  # TODO
                info_logger.reset()
                with torch.no_grad():
                    logs = {
                        **log_transition,
                        **log_infos,
                    }
                    logs.update({k: v.mean() for k, v in out_main.items()})
                wandb.log(
                    {
                        "speed": speed,
                        **logs,
                    },
                    step=global_step * args.num_envs,
                )

            if (
                args.checkpoint_interval > 0
                and global_step % args.checkpoint_interval == 0
                and start_time is not None
            ):
                save_params(
                    global_step=global_step,
                    actor=actor,
                    qnet=qnet,
                    qnet_target=qnet_target,
                    obs_normalizer=obs_normalizer,
                    critic_obs_normalizer=None,
                    args=args,
                    save_path=os.path.join(ckpt_dir, f"ckpt_{global_step}.pt"),
                )
            actor_scheduler.step()
            q_scheduler.step()

        if global_step >= args.measure_burnin:
            pbar.set_description(f"{speed: 4.4f} sps, " + desc)
    envs.close()
    wandb.finish()
    save_params(
        global_step=global_step,
        actor=actor,
        qnet=qnet,
        qnet_target=qnet_target,
        args=args,
        obs_normalizer=obs_normalizer,
        critic_obs_normalizer=None,
        save_path=os.path.join(ckpt_dir, f"ckpt_final.pt"),
    )


if __name__ == "__main__":
    try:
        if not args.log:
            os.environ["WANDB_MODE"] = "dryrun"
        main(args)
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print("Exception:", e)
    finally:
        simulation_app.close()
