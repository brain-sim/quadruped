# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
import tqdm
import wandb
from termcolor import colored

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaaclab.utils import configclass
from models import CNNPPOAgent, MLPPPOAgent
from utils import load_args, print_dict  # add load_args import
from hyperparams import EnvArgs, ExperimentPlayArgs
### TODO : Make play callable while training and after training.
### Solution - Use ManagerBasedRL or multi threading or multiprocessing to run train and eval.

### TODO : get the name of the video file from the environment.
### Solution - Use the environment's metadata to get the name of the video file (do it elegantly).


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


def make_isaaclab_env(
    task,
    device,
    num_envs,
    capture_video,
    disable_fabric,
    log_dir=None,
    video_length=200,
    *args,
    **kwargs,
):
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from wrappers import IsaacLabVecEnvWrapper

    import quadruped.tasks  # noqa: F401

    def thunk():
        cfg = parse_env_cfg(
            task, device, num_envs=num_envs, use_fabric=not disable_fabric
        )
        env = gym.make(
            task,
            cfg=cfg,
            render_mode="rgb_array"
            if (capture_video and log_dir is not None)
            else None,
            play_mode=True,
        )
        if capture_video and log_dir is not None:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        env = IsaacLabVecEnvWrapper(env)
        return env

    return thunk


def main(args):
    print_dict(args)
    run_name = f"{args.task}__{args.exp_name}"
    run = wandb.init(
        project="play",
        name=f"{os.path.splitext(os.path.basename(__file__))[0]}-{run_name}",
        config=vars(args),
        save_code=True,
    )
    run_dir = run.dir

    eval_envs = make_isaaclab_env(
        args.task,
        args.device,
        args.num_eval_envs,
        args.capture_video,
        args.disable_fabric,
        log_dir=run_dir,
    )()

    # Set camera eye position to (0, 0, 45) looking at origin
    print("📷 Setting camera eye position to (0, 0, 45)")
    eval_envs.unwrapped.sim.set_camera_view(eye=[0, 0, 60], target=[0.0, 0.0, 0.0])
    n_obs = int(np.prod(eval_envs.observation_space["policy"].shape[1:]))
    n_act = int(np.prod(eval_envs.action_space.shape[1:]))
    assert isinstance(eval_envs.action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )

    AGENT_LOOKUP = {
        "CNNPPOAgent": CNNPPOAgent,
        "MLPPPOAgent": MLPPPOAgent,
    }
    print("Here")

    agent_class = AGENT_LOOKUP[args.agent_type]
    agent = agent_class(n_obs, n_act)
    print(
        colored(
            "[INFO] : Loading agent from {args.checkpoint_path}",
            "green",
            attrs=["bold"],
        )
    )
    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )
    agent.load_state_dict(checkpoint)
    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )
    agent.to(device)
    agent.eval()

    @torch.no_grad()
    def run_play(play_agent, mode="play"):
        play_agent.eval()
        # environment evaluation
        eval_envs.seed(seed=args.seed)
        obs, _ = eval_envs.reset()
        step, global_step = 0, 0
        done = torch.zeros(args.num_eval_envs, dtype=torch.bool)

        # Create video directory
        video_dir = os.path.join(run_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)

        pbar = tqdm.tqdm(total=args.num_eval_env_steps)
        while step < args.num_eval_env_steps and not done.all():
            action = play_agent.get_action(obs)
            obs, _, done, _ = eval_envs.step(action)
            step += 1
            global_step += args.num_eval_envs
            pbar.update(1)
            pbar.set_description(f"step: {step}")

        eval_envs.close()

        # Log videos to wandb if capture_video is enabled
        if args.capture_video:
            for env_idx in range(args.num_eval_envs):
                video_path = os.path.join(video_dir, f"env_{env_idx}_fpp.mp4")
                wandb.log(
                    {
                        f"{mode}/env_{env_idx}_video": wandb.Video(
                            video_path,
                            format="mp4",
                            fps=30,
                        )
                    },
                    step=global_step,
                )

    run_play(agent)
    wandb.finish()


if __name__ == "__main__":
    try:
        os.environ["WANDB_MODE"] = "dryrun"
        main(args)
    except Exception as e:
        print("Exception:", e)
    finally:
        simulation_app.close()
