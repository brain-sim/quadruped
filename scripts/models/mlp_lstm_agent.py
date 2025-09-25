import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from ..utils import unpad_trajectories
from .base_agent import BaseAgent


class Memory(nn.Module):
    """Memory module for recurrent networks (rsl-rl style).

    Stores and manages hidden states of a GRU/LSTM core.
    """

    def __init__(
        self,
        input_size,
        type: str = "lstm",
        num_layers: int = 1,
        hidden_size: int = 256,
    ):
        super().__init__()
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            if hidden_states is None:
                raise ValueError(
                    "Hidden states not passed to memory module during policy update"
                )
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # Inference: use internal hidden state if none provided
            if hidden_states is None:
                out, self.hidden_states = self.rnn(
                    input.unsqueeze(0), self.hidden_states
                )
            else:
                out, self.hidden_states = self.rnn(input.unsqueeze(0), hidden_states)
        return out

    def step(self, input, hidden_states=None):
        # One step advance returning next hidden states for external control
        out, next_states = self.rnn(input.unsqueeze(0), hidden_states)
        return out.squeeze(0), next_states

    def reset(self, dones=None, hidden_states=None):
        if dones is None:  # reset all hidden states
            if hidden_states is None:
                self.hidden_states = None
            else:
                self.hidden_states = hidden_states
        elif self.hidden_states is not None:  # reset hidden states of done environments
            if hidden_states is None:
                if isinstance(self.hidden_states, tuple):  # LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = 0.0
                else:
                    self.hidden_states[..., dones == 1, :] = 0.0
            else:
                raise NotImplementedError(
                    "Resetting hidden states of done environments with custom hidden states is not implemented"
                )

    def detach_hidden_states(self, dones=None):
        if self.hidden_states is not None:
            if dones is None:  # detach all hidden states
                if isinstance(self.hidden_states, tuple):  # LSTM
                    self.hidden_states = tuple(
                        hidden_state.detach() for hidden_state in self.hidden_states
                    )
                else:
                    self.hidden_states = self.hidden_states.detach()
            else:  # detach hidden states of done environments
                if isinstance(self.hidden_states, tuple):  # LSTM
                    for hidden_state in self.hidden_states:
                        hidden_state[..., dones == 1, :] = hidden_state[
                            ..., dones == 1, :
                        ].detach()
                else:
                    self.hidden_states[..., dones == 1, :] = self.hidden_states[
                        ..., dones == 1, :
                    ].detach()


class MLPPPORecurrentAgent(BaseAgent):
    """
    Recurrent MLP PPO Agent matching rsl-rl ActorCriticRecurrentNetwork.

    Encoder MLP -> Memory (LSTM/GRU) -> actor mean head + critic value head.
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: type[nn.Module] = nn.ELU,
        layer_norm: bool = False,
        output_activation: type[nn.Module] | None = None,
        noise_std_type: str = "scalar",
        init_noise_std: float = 1.0,
        rnn_type: str = "lstm",
        rnn_hidden_size: int | None = None,
        rnn_num_layers: int = 1,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)

        self.n_obs = n_obs
        self.n_act = n_act
        self.noise_std_type = noise_std_type
        self.rnn_type = rnn_type.lower()
        self.rnn_num_layers = int(rnn_num_layers)
        self.rnn_hidden_size = int(rnn_hidden_size)

        # Memory (shared RNN core)
        self.memory_actor = Memory(
            input_size=n_obs,
            type=self.rnn_type,
            num_layers=self.rnn_num_layers,
            hidden_size=self.rnn_hidden_size,
        )

        self.memory_critic = Memory(
            input_size=n_obs,
            type=self.rnn_type,
            num_layers=self.rnn_num_layers,
            hidden_size=self.rnn_hidden_size,
        )

        # Heads
        self.actor = self.build_networks(
            input_dim=self.rnn_hidden_size,
            output_dim=n_act,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            output_activation=output_activation,
        )
        self.critic = self.build_networks(
            input_dim=self.rnn_hidden_size,
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
        )

        # Std parameterization
        if noise_std_type == "scalar":
            self.actor_std = nn.Parameter(init_noise_std * torch.ones(n_act))
        elif noise_std_type == "log":
            self.actor_std = nn.Parameter(torch.log(init_noise_std * torch.ones(n_act)))
        else:
            raise ValueError(f"Invalid noise_std_type: {noise_std_type}")

        Normal.set_default_validate_args(False)

        self.to(self.device, self.dtype)

    def reset(self, dones=None):
        self.memory_actor.reset(dones)
        self.memory_critic.reset(dones)

    def get_action(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute action from input."""
        out_memory = self.memory_actor(x, masks, hidden_states).squeeze(0)
        return self.actor(out_memory)

    def get_value(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute state-value from input."""
        out_memory = self.memory_critic(x, masks, hidden_states).squeeze(0)
        return self.critic(out_memory)

    def get_hidden_states(self):
        return self.memory_actor.hidden_states, self.memory_critic.hidden_states

    def get_action_and_value(
        self,
        obs,
        action: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
        hidden_states_actor: torch.Tensor | None = None,
        hidden_states_critic: torch.Tensor | None = None,
    ):
        if masks is None:
            if hidden_states_actor is None:
                out_memory_actor = self.memory_actor(obs).squeeze(0)
            if hidden_states_critic is None:
                out_memory_critic = self.memory_critic(obs).squeeze(0)
        else:
            out_memory_actor = self.memory_actor(
                obs, masks, hidden_states_actor
            ).squeeze(0)
            out_memory_critic = self.memory_critic(
                obs, masks, hidden_states_critic
            ).squeeze(0)
        action_mean = self.actor(out_memory_actor)
        action_std = self.actor_std.expand_as(action_mean)

        if self.noise_std_type == "log":
            action_std = torch.clamp(action_std, -20.0, 2.0)
            action_std = torch.exp(action_std)
        elif self.noise_std_type == "scalar":
            action_std = torch.clamp(action_std, min=1e-9)

        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action).sum(dim=-1),
            dist.entropy().sum(dim=-1),
            self.critic(out_memory_critic),
            action_mean,
            action_std,
        )
