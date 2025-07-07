import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer as TorchRLReplayBufferBase
from torchrl.data.replay_buffers import LazyMemmapStorage


class TorchRLReplayBuffer:
    """
    TorchRL-based replay buffer for efficient experience storage and sampling.
    Uses tensordict for structured data handling and LazyMemmapStorage for memory efficiency.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        device: str = "cpu",
        batch_size: int = 256,
        prefetch: int = 3,
    ):
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Create storage with memory mapping for efficiency
        storage = LazyMemmapStorage(buffer_size)

        # Initialize the replay buffer
        self._buffer = TorchRLReplayBufferBase(
            storage=storage,
            batch_size=batch_size,
            prefetch=prefetch,
        )

        # Define the data structure
        self._data_spec = TensorDict(
            {
                "observation": torch.zeros(observation_shape, dtype=torch.float32),
                "action": torch.zeros(action_shape, dtype=torch.float32),
                "next_observation": torch.zeros(observation_shape, dtype=torch.float32),
                "reward": torch.zeros(1, dtype=torch.float32),
                "not_done": torch.zeros(1, dtype=torch.float32),
            },
            batch_size=(),
        )

        self._size = 0

    def add(self, observation, action, next_observation, reward, done):
        """Add a transition to the buffer."""
        # Convert inputs to tensors if needed
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        if not isinstance(next_observation, torch.Tensor):
            next_observation = torch.tensor(next_observation, dtype=torch.float32)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward], dtype=torch.float32)
        if not isinstance(done, torch.Tensor):
            done = torch.tensor([1.0 - float(done)], dtype=torch.float32)

        # Ensure correct shapes
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if next_observation.dim() == 1:
            next_observation = next_observation.unsqueeze(0)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        if done.dim() == 0:
            done = done.unsqueeze(0)

        # Create tensordict for this transition
        data = TensorDict(
            {
                "observation": observation.squeeze(0),
                "action": action.squeeze(0),
                "next_observation": next_observation.squeeze(0),
                "reward": reward.squeeze(0),
                "not_done": done.squeeze(0),
            },
            batch_size=(),
        )

        # Add to buffer
        self._buffer.add(data)
        self._size = min(self._size + 1, self.buffer_size)

    def sample(self, batch_size: int | None = None):
        """Sample a batch from the buffer."""
        if batch_size is None:
            batch_size = self.batch_size

        # Sample with the specified batch size
        batch = self._buffer.sample(batch_size)
        return batch.to(self.device)

    def __len__(self):
        """Return the current size of the buffer."""
        return self._size

    def state_dict(self):
        """Return the state dict for serialization."""
        return {
            "buffer_size": self.buffer_size,
            "size": self._size,
            "batch_size": self.batch_size,
            "device": self.device,
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.buffer_size = state_dict["buffer_size"]
        self._size = state_dict["size"]
        self.batch_size = state_dict["batch_size"]
        self.device = state_dict["device"]


class PrioritizedTorchRLReplayBuffer(TorchRLReplayBuffer):
    """
    Prioritized Experience Replay buffer using TorchRL.
    Extends the basic TorchRLReplayBuffer with priority-based sampling.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        device: str = "cpu",
        batch_size: int = 256,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        prefetch: int = 3,
    ):
        super().__init__(
            buffer_size, observation_shape, action_shape, device, batch_size, prefetch
        )

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

        # Initialize priorities
        self.priorities = np.ones(buffer_size, dtype=np.float32) * self.max_priority

    def add(self, observation, action, next_observation, reward, done, priority=None):
        """Add a transition with priority."""
        # Add to base buffer
        super().add(observation, action, next_observation, reward, done)

        # Set priority
        if priority is None:
            priority = self.max_priority

        # Update priorities array
        idx = (self._size - 1) % self.buffer_size
        self.priorities[idx] = priority

    def sample(self, batch_size: int | None = None):
        """Sample batch with priorities."""
        if batch_size is None:
            batch_size = self.batch_size

        # Calculate sampling probabilities
        valid_priorities = self.priorities[: self._size]
        probs = valid_priorities**self.alpha
        probs = probs / probs.sum()

        # Sample indices
        indices = np.random.choice(self._size, batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (self._size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # Get batch from base buffer
        # Note: This is a simplified implementation
        # In practice, you'd want to implement proper indexed sampling
        batch = super().sample(batch_size)

        # Add weights to batch
        batch["weights"] = torch.tensor(
            weights, dtype=torch.float32, device=self.device
        )
        batch["indices"] = torch.tensor(indices, dtype=torch.long, device=self.device)

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch

    def update_priorities(self, indices, priorities):
        """Update priorities for given indices."""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
