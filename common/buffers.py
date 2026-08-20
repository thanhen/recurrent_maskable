from functools import partial
from typing import Generator, NamedTuple, Optional, Union, Callable, Tuple

import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.type_aliases import TensorDict


class RNNStates(NamedTuple):
    pi: Tuple[th.Tensor, ...]
    vf: Tuple[th.Tensor, ...]


class RecurrentMaskableRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor
    action_masks: th.Tensor


class RecurrentMaskableDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: RNNStates
    episode_starts: th.Tensor
    mask: th.Tensor
    action_masks: th.Tensor


# Window sizes the auto mode (max_seq_len < 0) climbs as episodes get longer.
SEQ_LEN_RUNGS = (16, 24, 32, 48, 64, 96, 128, 192, 256, 360)


def pad(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: th.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Chunk sequences and pad them to have constant dimensions.

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device
    :param tensor: Tensor of shape (batch_size, *tensor_shape)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq, max_length, *tensor_shape)
    """
    # Create sequences given start and end
    seq = [th.tensor(tensor[start : end + 1], device=device) for start, end in zip(seq_start_indices, seq_end_indices)]
    return th.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)


def pad_and_flatten(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: th.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Pad and flatten the sequences of scalar values,
    while keeping the sequence order.
    From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,)

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device (cpu, gpu, ...)
    :param tensor: Tensor of shape (max_length, n_seq, 1)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq * max_length,) aka (padded_batch_size,)
    """
    return pad(seq_start_indices, seq_end_indices, device, tensor, padding_value).flatten()


def pack_sequences(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    force_break: np.ndarray,
    max_seq_len: int,
    fill_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack consecutive episodes into windows of capacity ``max_seq_len``.

    Padding is what blows up memory: every window is padded to the longest one in the
    minibatch, so a handful of long episodes used to inflate the whole batch. Packing
    several short episodes into one window keeps the padded batch close to the real
    batch size whatever the episode length distribution looks like.

    A window is closed early (leaving padding) only when it is already at least
    ``1 - fill_ratio`` full; otherwise the next episode is split to fill it exactly.
    That bounds the padding ratio at ``1 / (1 - fill_ratio)`` while leaving short
    episodes intact as long as episodes stay much shorter than ``max_seq_len``.

    Windows never cross an env boundary and always tile the input contiguously,
    so no transition is dropped or duplicated.
    """
    min_fill = int(max_seq_len * (1.0 - fill_ratio))
    new_starts: list = []
    new_ends: list = []
    cur_start = 0
    cur_len = 0

    def close_window():
        nonlocal cur_len
        if cur_len > 0:
            new_starts.append(cur_start)
            new_ends.append(cur_start + cur_len - 1)
            cur_len = 0

    for i in range(len(seq_start_indices)):
        if force_break[i]:
            close_window()
        pos = int(seq_start_indices[i])
        remaining = int(seq_end_indices[i]) - pos + 1
        while remaining > 0:
            if cur_len == 0:
                cur_start = pos
            space = max_seq_len - cur_len
            if remaining <= space:
                cur_len += remaining
                pos += remaining
                remaining = 0
                if cur_len == max_seq_len:
                    close_window()
            elif cur_len < min_fill:
                cur_len += space
                pos += space
                remaining -= space
                close_window()
            else:
                # Already full enough: closing wastes less than splitting the episode.
                close_window()
    close_window()

    return (
        np.array(new_starts, dtype=np.int64),
        np.array(new_ends, dtype=np.int64),
    )


def create_sequencers(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: th.device,
    max_seq_len: int = 0,
    fill_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, Callable, Callable]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.

    :param episode_starts: Indices where an episode starts
    :param env_change: Indices where the data collected
        come from a different env (when using multiple env for data collection)
    :param device: PyTorch device
    :param max_seq_len: Window capacity for packing. 0 disables packing
        (one sequence per episode, original behavior).
    :param fill_ratio: Max fraction of a window left as padding before an episode
        gets split to fill it.
    :return: Indices of the transitions that start/end a sequence,
        pad and pad_and_flatten utilities tailored for this batch
        (sequence starts and ends indices are fixed)
    """
    # Create sequence if env changes too
    seq_start = np.logical_or(episode_starts, env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    # Retrieve indices of sequence starts
    seq_start_indices = np.where(seq_start == True)[0]  # noqa: E712
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts) - 1])])

    if max_seq_len > 0:
        force_break = np.asarray(env_change).flatten()[seq_start_indices] > 0
        force_break[0] = True
        seq_start_indices, seq_end_indices = pack_sequences(
            seq_start_indices, seq_end_indices, force_break, max_seq_len, fill_ratio
        )

    # Create padding method for this minibatch
    # to avoid repeating arguments (seq_start_indices, seq_end_indices)
    local_pad = partial(pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(pad_and_flatten, seq_start_indices, seq_end_indices, device)
    return seq_start_indices, seq_end_indices, local_pad, local_pad_and_flatten


class RecurrentMaskableRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the LSTM cell and hidden states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
        (n_steps, lstm.num_layers, n_envs, lstm.hidden_size)
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        max_seq_len: int = 0,
        seq_len_min: int = 32,
        seq_len_cap: int = 128,
        seq_len_fill_ratio: float = 0.15,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        self._buffers_allocated = False
        self.max_seq_len = max_seq_len
        self.seq_len_min = seq_len_min
        self.seq_len_cap = seq_len_cap
        self.seq_len_fill_ratio = seq_len_fill_ratio
        # Auto mode (max_seq_len < 0) climbs these rungs as episodes get longer.
        self._auto_seq_len = seq_len_min
        self._auto_over_count = 0
        self._seq_len_percentiles = {}
        self._reset_pad_stats()
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    @property
    def effective_seq_len(self) -> int:
        return self._auto_seq_len if self.max_seq_len < 0 else self.max_seq_len

    def _reset_pad_stats(self) -> None:
        self._pad_stats = {"padded": 0, "real": 0, "n_seq": 0, "calls": 0, "ratio_max": 0.0}

    def raw_seq_lengths(self) -> np.ndarray:
        """Episode lengths over the whole rollout, per env, before any packing."""
        lengths = []
        for env_idx in range(self.n_envs):
            starts = np.flatnonzero(self.episode_starts[:, env_idx])
            if starts.size == 0 or starts[0] != 0:
                starts = np.concatenate(([0], starts))
            ends = np.concatenate((starts[1:] - 1, [self.buffer_size - 1]))
            lengths.append(ends - starts + 1)
        return np.concatenate(lengths)

    def update_auto_seq_len(self) -> None:
        """Recompute sequence-length stats once per rollout, and grow the auto window."""
        lengths = self.raw_seq_lengths()
        self._seq_len_percentiles = {
            "p50": float(np.percentile(lengths, 50)),
            "p90": float(np.percentile(lengths, 90)),
            "p99": float(np.percentile(lengths, 99)),
            "max": float(lengths.max()),
        }
        if self.max_seq_len >= 0:
            return
        p99 = self._seq_len_percentiles["p99"]
        if p99 > self._auto_seq_len and self._auto_seq_len < self.seq_len_cap:
            self._auto_over_count += 1
        else:
            self._auto_over_count = 0
        # Require several rollouts in a row before growing: changing the window changes
        # both throughput and where BPTT is truncated.
        if self._auto_over_count >= 3:
            for rung in SEQ_LEN_RUNGS:
                if rung > self._auto_seq_len and rung <= self.seq_len_cap:
                    self._auto_seq_len = rung
                    break
            self._auto_over_count = 0

    def get_seq_stats(self) -> dict:
        stats = dict(self._seq_len_percentiles)
        s = self._pad_stats
        if s["calls"] > 0:
            stats["pad_ratio_mean"] = s["padded"] / max(s["real"], 1)
            stats["pad_ratio_max"] = s["ratio_max"]
            stats["n_seq"] = s["n_seq"] / s["calls"]
        stats["max_seq_len_eff"] = float(self.effective_seq_len)
        return stats

    def reset(self):
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims

        # Reuse existing arrays to avoid heap fragmentation from repeated alloc/free.
        # After get() is called, arrays are reshaped by swap_and_flatten so check shape.
        obs_shape_expected = (self.buffer_size, self.n_envs, *self.obs_shape)
        can_reuse = (
            self._buffers_allocated
            and hasattr(self, 'observations')
            and self.observations.shape == obs_shape_expected
        )

        if can_reuse:
            self.action_masks[:] = 1.0
            self.hidden_states_pi[:] = 0
            self.cell_states_pi[:] = 0
            self.hidden_states_vf[:] = 0
            self.cell_states_vf[:] = 0
            # Reuse parent buffer arrays (observations, actions, rewards, etc.)
            self.observations[:] = 0
            self.actions[:] = 0
            self.rewards[:] = 0
            self.returns[:] = 0
            self.episode_starts[:] = 0
            self.values[:] = 0
            self.log_probs[:] = 0
            self.advantages[:] = 0
            # Free temporary flattened copies from previous get()
            for key in list(self.__dict__):
                if key.startswith('_flat_'):
                    del self.__dict__[key]
            self._reset_pad_stats()
            self.generator_ready = False
            # BaseBuffer.reset()
            self.pos = 0
            self.full = False
        else:
            self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)
            super().reset()
            self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
            self.cell_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
            self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
            self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
            self._buffers_allocated = True
            self._reset_pad_stats()


    def add(self, *args, lstm_states: RNNStates, action_masks: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        :param hidden_states: LSTM cell and hidden state
        """
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))
    
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

        super().add(*args, **kwargs)


    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentMaskableRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data — work on copies to preserve originals for reuse
        if not self.generator_ready:
            # observations and the LSTM states are NOT flattened here: a flat copy would
            # double the largest arrays in the buffer, and _get_samples only ever reads
            # n_seq rows of the states and one window at a time of the observations.
            # flatten but keep the sequence order
            # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
            # 2. (n_envs, n_steps, *tensor_shape) -> (n_envs * n_steps, *tensor_shape)
            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "episode_starts",
                "action_masks",
            ]:
                self.__dict__[f"_flat_{tensor}"] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _pad_observations(self, batch_inds: np.ndarray, max_length: int) -> th.Tensor:
        """
        Build the padded observation tensor straight from the rollout array.

        Observations dominate the buffer (2.6 GB at n_steps=4096, n_envs=8, obs_dim=20040),
        so a flattened copy would double the master process footprint for no benefit:
        every window is copied into the padded tensor one at a time anyway.
        """
        step_idx = batch_inds % self.buffer_size
        env_idx = batch_inds // self.buffer_size
        out = th.zeros(
            (len(self.seq_start_indices), max_length, *self.obs_shape),
            dtype=th.float32,
            device=self.device,
        )
        for i, (start, end) in enumerate(zip(self.seq_start_indices, self.seq_end_indices)):
            window = self.observations[step_idx[start:end + 1], env_idx[start:end + 1]]
            # copy_ writes host -> device straight into the slice, no staging tensor
            out[i, : end - start + 1].copy_(th.from_numpy(window))
        return out

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentMaskableRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.seq_end_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self._flat_episode_starts[batch_inds], env_change[batch_inds], self.device,
            max_seq_len=self.effective_seq_len,
            fill_ratio=self.seq_len_fill_ratio,
        )

        # Number of sequences
        n_seq = len(self.seq_start_indices)
        max_length = int((self.seq_end_indices - self.seq_start_indices).max()) + 1
        padded_batch_size = n_seq * max_length

        pad_ratio = padded_batch_size / len(batch_inds)
        stats = self._pad_stats
        stats["padded"] += padded_batch_size
        stats["real"] += len(batch_inds)
        stats["n_seq"] += n_seq
        stats["calls"] += 1
        stats["ratio_max"] = max(stats["ratio_max"], pad_ratio)

        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence.
        # swap_and_flatten lays the buffer out env-major, so a flat index maps back to
        # (step, env) and the original (n_steps, n_layers, n_envs, dim) arrays can be
        # indexed directly instead of keeping a flattened copy of the whole rollout.
        flat_idx = batch_inds[self.seq_start_indices]
        step_idx = flat_idx % self.buffer_size
        env_idx = flat_idx // self.buffer_size
        lstm_states_pi = (
            # (n_steps, n_envs, n_layers, dim) -> (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi.swapaxes(1, 2)[step_idx, env_idx].swapaxes(0, 1),
            self.cell_states_pi.swapaxes(1, 2)[step_idx, env_idx].swapaxes(0, 1),
        )
        lstm_states_vf = (
            self.hidden_states_vf.swapaxes(1, 2)[step_idx, env_idx].swapaxes(0, 1),
            self.cell_states_vf.swapaxes(1, 2)[step_idx, env_idx].swapaxes(0, 1),
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())

        return RecurrentMaskableRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=self._pad_observations(batch_inds, max_length).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(self._flat_actions[batch_inds]).reshape((padded_batch_size,) + self._flat_actions.shape[1:]),
            old_values=self.pad_and_flatten(self._flat_values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self._flat_log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self._flat_advantages[batch_inds]),
            returns=self.pad_and_flatten(self._flat_returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self._flat_episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self._flat_returns[batch_inds])),
            action_masks=self.pad(self._flat_action_masks[batch_inds]).reshape((padded_batch_size,) + self._flat_action_masks.shape[1:])
        )


class RecurrentMaskableDictRolloutBuffer(DictRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RecurrentRolloutBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect lstm states
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        max_seq_len: int = 0,
        seq_len_fill_ratio: float = 0.15,
    ):
        self.action_masks = None
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        self.max_seq_len = max_seq_len
        self.seq_len_fill_ratio = seq_len_fill_ratio
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs)

    def reset(self):
        if isinstance(self.action_space, spaces.Discrete):
            mask_dims = self.action_space.n
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            mask_dims = sum(self.action_space.nvec)
        elif isinstance(self.action_space, spaces.MultiBinary):
            mask_dims = 2 * self.action_space.n  # One mask per binary outcome
        else:
            raise ValueError(f"Unsupported action space {type(self.action_space)}")

        self.mask_dims = mask_dims
        self.action_masks = np.ones((self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32)

        super().reset()
        self.hidden_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.hidden_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.cell_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)

    def add(self, *args, lstm_states: RNNStates, action_masks: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        :param hidden_states: LSTM cell and hidden state
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0].cpu().numpy())
        self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1].cpu().numpy())
        self.hidden_states_vf[self.pos] = np.array(lstm_states.vf[0].cpu().numpy())
        self.cell_states_vf[self.pos] = np.array(lstm_states.vf[1].cpu().numpy())

        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape((self.n_envs, self.mask_dims))

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentMaskableDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)
            # swap first to (self.n_steps, self.n_envs, lstm.num_layers, lstm.hidden_size)
            for tensor in ["hidden_states_pi", "cell_states_pi", "hidden_states_vf", "cell_states_vf"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
                "action_masks",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size


    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentMaskableDictRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.seq_end_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device,
            max_seq_len=self.max_seq_len,
            fill_ratio=self.seq_len_fill_ratio,
        )

        n_seq = len(self.seq_start_indices)
        max_length = int((self.seq_end_indices - self.seq_start_indices).max()) + 1
        padded_batch_size = n_seq * max_length
        # We retrieve the lstm hidden states that will allow
        # to properly initialize the LSTM at the beginning of each sequence
        lstm_states_pi = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.hidden_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
            self.cell_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        lstm_states_pi = (self.to_torch(lstm_states_pi[0]).contiguous(), self.to_torch(lstm_states_pi[1]).contiguous())
        lstm_states_vf = (self.to_torch(lstm_states_vf[0]).contiguous(), self.to_torch(lstm_states_vf[1]).contiguous())

        observations = {key: self.pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        observations = {key: obs.reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in observations.items()}

        return RecurrentMaskableDictRolloutBufferSamples(
            observations=observations,
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=RNNStates(lstm_states_pi, lstm_states_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
            action_masks=self.pad(self.action_masks[batch_inds]).reshape((padded_batch_size,) + self.action_masks.shape[1:])
        )
