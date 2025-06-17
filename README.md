# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import Isaac's Spot configuration to extend it
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.flat_env_cfg import SpotFlatEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# CRUCIAL: Import the Spot robot configuration
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

# Import our local MDP functions
from . import mdp as local_mdp

# Template for Isaac Lab Projects

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/quadruped

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/quadruped/quadruped/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/quadruped"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```

# Quadruped Obstacle Navigation

This package provides Spot quadruped environments for obstacle navigation tasks, built on top of Isaac Lab's robust Spot configurations.

## Environments

The package includes several environments that extend Isaac Lab's `Isaac-Velocity-Flat-Spot-v0` configuration with obstacle navigation capabilities:

### Core Environments

1. **`Spot-Velocity-Flat-Nav-v0`**
   - Basic Spot velocity tracking on flat terrain
   - Extends Isaac's Spot configuration with longer episodes
   - Good baseline for comparison

2. **`Spot-Velocity-Flat-Obstacle-Nav-v0`**
   - Spot velocity tracking with cuboid obstacles
   - Obstacles are randomly placed in the environment
   - Rewards for climbing over obstacles and maintaining forward progress
   - 3-8 obstacles per environment, randomly positioned

3. **`Spot-Velocity-Rough-Obstacle-Nav-v0`**
   - Same as flat obstacle navigation but on rough terrain
   - Combines obstacle climbing with terrain traversal challenges

### Play Environments (for Testing)

4. **`Spot-Velocity-Flat-Obstacle-Nav-Play-v0`**
   - Fewer environments (16 vs 1024) for better visualization
   - Longer episodes (60s vs 30s) for extended testing
   - Reduced obstacle randomization for predictable scenarios

5. **`Spot-Velocity-Rough-Obstacle-Nav-Play-v0`**
   - Play version of rough obstacle navigation

## Features

### Obstacle Configuration
- **Cuboid obstacles**: 0.4m × 0.8m × 0.3m (width × length × height)
- **Physics**: Stable with appropriate friction (static: 0.8, dynamic: 0.6)
- **Positioning**: Randomized between 1-6m ahead, ±2m laterally
- **Count**: 3-8 obstacles per environment (randomized on reset)

### Reward System
The environments add obstacle-specific rewards to Isaac's base Spot rewards:

1. **Obstacle Clearance Reward** (weight: 5.0)
   - Encourages maintaining height above obstacles when nearby
   - Triggers when robot is within 1m of an obstacle
   - Rewards height clearance above 0.15m threshold

2. **Forward Progress Reward** (weight: 2.0)
   - Rewards forward velocity, especially after overcoming obstacles
   - Clamped to 0-3 m/s range

### Key Design Principles
- **Minimal Code**: Extends Isaac's proven Spot configurations
- **Stable Obstacles**: Physics tuned for reliable climbing behavior
- **Adaptive Rewards**: Encourages both obstacle avoidance and climbing
- **Scalable**: Works with 16-1024 parallel environments

## Usage

### Testing Environments

```bash
# Test all environments
cd quadruped
python scripts/test_obstacle_nav.py --env all --episodes 3

# Test specific environment
python scripts/test_obstacle_nav.py --env flat-obs --episodes 5

# Test play version for visualization
python scripts/test_obstacle_nav.py --env play-flat --episodes 2
```

### Using in Training

```python
import gymnasium as gym
import quadruped.tasks.manager_based.quadruped  # Register environments

# Create environment
env = gym.make("Spot-Velocity-Flat-Obstacle-Nav-v0")

# Or create play version for testing
env = gym.make("Spot-Velocity-Flat-Obstacle-Nav-Play-v0")
```

### Environment Parameters

| Parameter | Training Envs | Play Envs |
|-----------|---------------|-----------|
| Number of environments | 1024 | 16 |
| Environment spacing | 8.0m | 10.0m |
| Episode length | 30s | 60s |
| Obstacle count | 3-8 | 2-4 |
| Decimation | 10 (50Hz) | 10 (50Hz) |

## Dependencies

- Isaac Lab
- Isaac Sim
- PyTorch
- Gymnasium

## Implementation Details

The environments are implemented by:
1. Importing Isaac's `SpotFlatEnvCfg` 
2. Adding `RigidObjectCfg` for cuboid obstacles
3. Implementing custom reward functions in `mdp/rewards.py`
4. Adding obstacle randomization events
5. Minimal configuration override for different variants

This approach maximizes code reuse while providing robust obstacle navigation capabilities.