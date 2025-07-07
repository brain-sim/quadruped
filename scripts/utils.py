import os
import random

import jax
import numpy as np
import torch

# TorchRL imports
from termcolor import colored


def seed_everything(
    envs,
    seed,
    use_torch=False,
    use_jax=False,
    torch_deterministic=False,
):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if isinstance(envs, list):
        for env in envs:
            env.seed(seed=seed)
    else:
        envs.seed(seed=seed)
    if use_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
    if use_jax:
        key = jax.random.PRNGKey(seed)
        return key


def set_high_precision():
    torch.set_float32_matmul_precision("high")


def update_learning_rate_adaptive(
    optimizer, kl_divergence, desired_kl, lr_multiplier, min_lr=1e-6, max_lr=1e-2
):
    current_lr = optimizer.param_groups[0]["lr"]

    if kl_divergence > desired_kl * 2.0:
        new_lr = current_lr / lr_multiplier
    elif kl_divergence < desired_kl / 2.0 and kl_divergence > 0.0:
        new_lr = current_lr * lr_multiplier
    else:
        new_lr = current_lr

    # Clamp learning rate to reasonable bounds
    new_lr = np.clip(new_lr, min_lr, max_lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

    return new_lr


def print_dict(
    val,
    nesting: int = -4,
    start: bool = True,
    color: str = None,
    attrs: list[str] = None,
):
    """Outputs a nested dictionary."""
    if isinstance(val, dict):
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(colored(k, color, attrs=attrs), end=": ")
            print_dict(val[k], nesting, start=False)
    else:
        # deal with functions in print statements
        if callable(val):
            print(colored(callable_to_string(val), color, attrs=attrs))
        else:
            print(colored(val, color, attrs=attrs))


def callable_to_string(obj):
    """Convert a callable to a string representation."""
    if hasattr(obj, "__name__"):
        return obj.__name__
    else:
        return str(obj)
