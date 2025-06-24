import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class MLPPPOAgent(nn.Module):
    def __init__(
        self,
        n_obs,
        n_act,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation=nn.ELU,
        noise_std_type="scalar",
        init_noise_std=1.0,
    ):
        super().__init__()
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.actor_std = nn.Parameter(init_noise_std * torch.ones(n_act))
        elif noise_std_type == "log":
            self.actor_std = nn.Parameter(torch.log(init_noise_std * torch.ones(n_act)))
        else:
            raise ValueError(f"Invalid noise_std_type: {noise_std_type}")
        critic_layers = []
        actor_layers = []
        for i in range(len(actor_hidden_dims)):
            if i == 0:
                actor_layers.append(nn.Linear(n_obs, actor_hidden_dims[i]))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[i - 1], actor_hidden_dims[i])
                )
            actor_layers.append(activation())
        for i in range(len(critic_hidden_dims)):
            if i == 0:
                critic_layers.append(nn.Linear(n_obs, critic_hidden_dims[i]))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[i - 1], critic_hidden_dims[i])
                )
            critic_layers.append(activation())
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], n_act))
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.actor = nn.Sequential(*actor_layers)
        self.critic = nn.Sequential(*critic_layers)
        Normal.set_default_validate_args(False)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None, eval_mode=False):
        action_mean = self.actor(obs)
        action_std = self.actor_std.expand_as(action_mean)
        if self.noise_std_type == "log":
            action_std = torch.clamp(action_std, -20.0, 2.0)
            action_std = torch.exp(action_std)
        dist = Normal(action_mean, action_std)
        if eval_mode:
            action = action_mean
        elif action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(dim=-1),
            dist.entropy().sum(dim=-1),
            self.critic(obs),
        )

    def get_action(self, obs):
        action_mean = self.actor(obs)
        return action_mean

    def forward(self, obs):
        return self.get_action(obs)


# Load and convert your model
def convert_checkpoint_to_jit(n_obs, n_act, checkpoint_path, output_path):
    # Initialize model
    model = MLPPPOAgent(n_obs=n_obs, n_act=n_act)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load state dict (adjust key name as needed)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)  # Assume checkpoint is the state_dict

    model.eval()

    # Create example input (batch_size=1, obs_dim=48 for Spot)
    example_input = torch.randn(1, n_obs)

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    # Save the traced model
    traced_model.save(output_path)
    print(f"JIT model saved to {output_path}")

    return traced_model


# Usage
if __name__ == "__main__":
    n_obs = 48
    n_act = 12
    traced_model = convert_checkpoint_to_jit(
        n_obs,
        n_act,
        "/home/chandramouli/quadruped/wandb/run-20250624_161703-yq15s6c0/files/checkpoints/ckpt_58982400.pt",
        "/home/chandramouli/cognitiverl/source/cognitiverl/cognitiverl/tasks/direct/custom_assets/spot_policy_v5.pt",
    )

    # Test the traced model
    test_input = torch.zeros(1, n_obs)
    test_input[:, 8] = -1.0
    output = traced_model(test_input)
    print(output)
    print(f"Test output shape: {output.shape}")
