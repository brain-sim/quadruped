import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

def custom_height_scan(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.0
) -> torch.Tensor:
    """Custom height scan function that returns the height of the robot's body."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return torch.clamp(height, 0.0, 1e6)

