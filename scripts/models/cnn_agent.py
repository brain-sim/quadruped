import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .base_agent import BaseAgent


class CNNPPOAgent(BaseAgent):
    """
    CNN PPO Agent using MobileNetV3 backbone.
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        img_size: list[int] = [3, 32, 32],
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: type[nn.Module] = nn.ELU,
        noise_std_type: str = "scalar",
        init_noise_std: float = 1.0,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)

        self.n_obs = n_obs
        self.n_act = n_act
        self.noise_std_type = noise_std_type

        # Image dimensions
        channels, height, width = img_size
        self.img_size = (channels, height, width)

        # Setup backbone and get feature size
        self._setup_backbone(channels)

        # Build networks using base class method
        self.actor = self.build_networks(
            input_dim=self.feature_size,
            output_dim=n_act,
            hidden_dims=actor_hidden_dims,
            activation=activation,
        )

        self.critic = self.build_networks(
            input_dim=self.feature_size,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        # Initialize noise parameters
        if noise_std_type == "scalar":
            self.actor_std = nn.Parameter(init_noise_std * torch.ones(n_act))
        elif noise_std_type == "log":
            self.actor_std = nn.Parameter(torch.log(init_noise_std * torch.ones(n_act)))
        else:
            raise ValueError(f"Invalid noise_std_type: {noise_std_type}")

        Normal.set_default_validate_args(False)

        # Move to device and set precision
        self.to(self.device, self.dtype)

    def _setup_backbone(self, channels: int):
        """Setup MobileNetV3 backbone."""
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone.eval()

        # Adjust first conv layer
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=channels,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.backbone = nn.Sequential(*list(self.backbone.features))

        # Get feature size
        with torch.no_grad():
            dummy = torch.zeros(1, *self.img_size)
            self.feature_size = self.backbone(dummy).view(1, -1).size(1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from image input."""
        batch_size = x.size(0)
        c, h, w = self.img_size

        # Reshape to image format
        imgs = x[:, : c * h * w].view(batch_size, c, h, w)

        # Extract features
        with torch.no_grad():
            features = self.backbone(imgs)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state-value from raw input."""
        img_feats = self.extract_image(x)
        return self.critic(img_feats)

    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action from input."""
        features = self.extract_features(x)
        return self.actor(features)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state-value from input."""
        features = self.extract_features(x)
        return self.critic(features)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple:
        """Compute action, log-prob, entropy, and value."""
        features = self.extract_features(x)
        action_mean = self.actor(features)
        action_std = self.actor_std.expand_as(action_mean)

        if self.noise_std_type == "log":
            action_std = torch.clamp(action_std, -20.0, 5.0)
            action_std = torch.exp(action_std)
        elif self.noise_std_type == "scalar":
            action_std = torch.clamp(action_std, min=1e-6)

        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action).sum(dim=-1),
            dist.entropy().sum(dim=-1),
            self.critic(features),
            action_mean,
            action_std,
        )

    def forward(self, obs):
        return self.get_action(obs)


class CNNTD3Actor:
    pass


class CNNTD3Critic:
    pass


class CNNFastTD3Actor:
    pass


class CNNFastTD3Critic:
    pass