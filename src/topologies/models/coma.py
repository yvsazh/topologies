import copy
import math

import torch
from torch import nn


class COMA(nn.Module):
    architecture_id = "coma"
    PRESETS = {
        "tiny": {
            "num_agents": 2,
            "observation_dim": 4,
            "state_dim": 8,
            "num_actions": 3,
            "actor_hidden_size": 16,
            "critic_hidden_sizes": (16, 16),
        },
        "small": {
            "num_agents": 5,
            "observation_dim": 16,
            "state_dim": 32,
            "num_actions": 8,
            "actor_hidden_size": 64,
            "critic_hidden_sizes": (64, 64),
        },
        "base": {
            "num_agents": 5,
            "observation_dim": 32,
            "state_dim": 64,
            "num_actions": 12,
            "actor_hidden_size": 128,
            "critic_hidden_sizes": (128, 128),
        },
    }

    def __init__(
        self,
        num_agents,
        observation_dim,
        state_dim,
        num_actions,
        actor_hidden_size=128,
        critic_hidden_sizes=(128, 128),
        epsilon_start=0.5,
        epsilon_end=0.02,
        epsilon_anneal_episodes=750,
        gamma=0.99,
        td_lambda=0.8,
        critic_target_update_interval=150,
    ):
        super().__init__()
        dimensions = {
            "num_agents": num_agents,
            "observation_dim": observation_dim,
            "state_dim": state_dim,
            "num_actions": num_actions,
            "actor_hidden_size": actor_hidden_size,
            "epsilon_anneal_episodes": epsilon_anneal_episodes,
            "critic_target_update_interval": critic_target_update_interval,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(critic_hidden_sizes, (tuple, list)) or not critic_hidden_sizes
                or any(isinstance(width, bool) or not isinstance(width, int) or width <= 0
                       for width in critic_hidden_sizes)):
            raise ValueError("critic_hidden_sizes must contain positive integers")
        for name, value in (("epsilon_start", epsilon_start), ("epsilon_end", epsilon_end)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        for name, value in (("gamma", gamma), ("td_lambda", td_lambda)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be a finite number in [0, 1]")

        self.num_agents = num_agents
        self.observation_dim = observation_dim
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_sizes = tuple(critic_hidden_sizes)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_anneal_episodes = epsilon_anneal_episodes
        self.gamma = float(gamma)
        self.td_lambda = float(td_lambda)
        self.critic_target_update_interval = critic_target_update_interval

        actor_input_size = observation_dim + num_actions + num_agents
        self.actor = nn.ModuleDict({
            "input": nn.Linear(actor_input_size, actor_hidden_size),
            "gru": nn.GRU(actor_hidden_size, actor_hidden_size, batch_first=True),
            "output": nn.Linear(actor_hidden_size, num_actions),
        })
        critic_input_size = state_dim + num_agents * observation_dim + num_agents * num_actions + num_agents
        critic_layers = []
        current_size = critic_input_size
        for width in self.critic_hidden_sizes:
            critic_layers.extend((nn.Linear(current_size, width), nn.ReLU()))
            current_size = width
        critic_layers.append(nn.Linear(current_size, num_actions))
        self.critic = nn.Sequential(*critic_layers)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.requires_grad_(False)
        self.target_critic.eval()

    @staticmethod
    def _validate_actions(actions, num_actions, expected_shape=None):
        if not isinstance(actions, torch.Tensor):
            COMA._raise_value_error("actions must be a torch tensor")
        if actions.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("actions must contain integer action indices")
        if expected_shape is not None and tuple(actions.shape) != tuple(expected_shape):
            raise ValueError("actions shape must match the leading Q-value dimensions")
        if actions.numel() and (torch.any(actions < 0) or torch.any(actions >= num_actions)):
            raise ValueError("action indices must be in the configured range")

    @staticmethod
    def _raise_value_error(message):
        raise ValueError(message)

    @staticmethod
    def bounded_softmax(logits, epsilon):
        if not isinstance(logits, torch.Tensor) or logits.ndim < 1 or not logits.is_floating_point():
            raise ValueError("logits must be a floating-point tensor with an action dimension")
        if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be a finite number in [0, 1]")
        return (1.0 - epsilon) * torch.softmax(logits, dim=-1) + epsilon / logits.shape[-1]

    @staticmethod
    def counterfactual_advantage(q_values, probabilities, actions):
        if (not isinstance(q_values, torch.Tensor) or q_values.ndim < 2
                or not q_values.is_floating_point()):
            raise ValueError("q_values must be floating-point with agent and action dimensions")
        if (not isinstance(probabilities, torch.Tensor) or probabilities.shape != q_values.shape
                or not probabilities.is_floating_point()):
            raise ValueError("probabilities must be floating-point and match q_values")
        COMA._validate_actions(actions, q_values.shape[-1], q_values.shape[:-1])
        selected_q = q_values.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        baseline = (q_values * probabilities).sum(dim=-1)
        return selected_q - baseline

    @staticmethod
    def actor_loss(probabilities, actions, advantages):
        if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim < 1
                or not probabilities.is_floating_point()):
            raise ValueError("probabilities must be a floating-point tensor with an action dimension")
        COMA._validate_actions(actions, probabilities.shape[-1], probabilities.shape[:-1])
        if (not isinstance(advantages, torch.Tensor) or advantages.shape != actions.shape
                or not advantages.is_floating_point()):
            raise ValueError("advantages must be floating-point and match actions")
        selected = probabilities.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        selected = selected.clamp_min(torch.finfo(selected.dtype).tiny)
        return -(selected.log() * advantages.detach()).mean()

    @staticmethod
    def critic_loss(q_values, actions, targets):
        if (not isinstance(q_values, torch.Tensor) or q_values.ndim < 2
                or not q_values.is_floating_point()):
            raise ValueError("q_values must be floating-point with an action dimension")
        COMA._validate_actions(actions, q_values.shape[-1], q_values.shape[:-1])
        if (not isinstance(targets, torch.Tensor) or targets.shape != actions.shape
                or not targets.is_floating_point()):
            raise ValueError("targets must be floating-point and match actions")
        selected_q = q_values.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        return (selected_q - targets).square().mean()

    def td_lambda_targets(self, rewards, next_values, terminals, gamma=None, td_lambda=None):
        if (not isinstance(rewards, torch.Tensor) or rewards.ndim < 1
                or not rewards.is_floating_point()):
            raise ValueError("rewards must be a floating-point tensor with a time dimension")
        if (not isinstance(next_values, torch.Tensor) or next_values.shape != rewards.shape
                or not next_values.is_floating_point()):
            raise ValueError("next_values must be floating-point and match rewards")
        if not isinstance(terminals, torch.Tensor):
            COMA._raise_value_error("terminals must be a tensor")
        gamma = self.gamma if gamma is None else gamma
        td_lambda = self.td_lambda if td_lambda is None else td_lambda
        for name, value in (("gamma", gamma), ("td_lambda", td_lambda)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        if terminals.dtype != torch.bool:
            if (not terminals.is_floating_point() and terminals.dtype not in
                    (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
                raise ValueError("terminals must contain boolean or binary values")
            if torch.any((terminals != 0) & (terminals != 1)):
                raise ValueError("terminals must contain only binary values")
        if terminals.ndim > rewards.ndim:
            raise ValueError("terminals must be broadcastable to rewards")
        while terminals.ndim < rewards.ndim:
            terminals = terminals.unsqueeze(-1)
        try:
            terminal_mask = torch.broadcast_to(terminals.bool(), rewards.shape)
        except RuntimeError as error:
            raise ValueError("terminals must be broadcastable to rewards") from error
        if rewards.ndim > 1:
            if rewards.shape[1] == 0:
                raise ValueError("rewards time dimension must be non-empty")
            sequence_dim = 1
        else:
            if rewards.shape[0] == 0:
                raise ValueError("rewards time dimension must be non-empty")
            sequence_dim = 0
        targets = [None] * rewards.shape[sequence_dim]
        next_return = None
        for step in range(len(targets) - 1, -1, -1):
            reward = rewards.select(sequence_dim, step)
            bootstrap = next_values.select(sequence_dim, step)
            terminal = terminal_mask.select(sequence_dim, step)
            continuation = bootstrap if next_return is None else (
                (1.0 - td_lambda) * bootstrap + td_lambda * next_return
            )
            next_return = reward + gamma * (~terminal).to(rewards.dtype) * continuation
            targets[step] = next_return
        return torch.stack(targets, dim=sequence_dim)

    def update_target_critic(self):
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

    @torch.no_grad()
    def target_q_values(self, state, observations, actions):
        if not isinstance(state, torch.Tensor) or state.ndim != 3:
            raise ValueError("state must have shape [batch, time, state_dim]")
        if not isinstance(observations, torch.Tensor) or observations.ndim != 4:
            raise ValueError("observations must have shape [batch, time, agents, observation_dim]")
        if (not state.is_floating_point() or not observations.is_floating_point()
                or state.device != observations.device or state.dtype != observations.dtype):
            raise ValueError("state and observations must be floating-point with matching device and dtype")
        batch_size, time_steps, agents, observation_dim = observations.shape
        if (batch_size <= 0 or time_steps <= 0 or agents != self.num_agents
                or observation_dim != self.observation_dim
                or tuple(state.shape) != (batch_size, time_steps, self.state_dim)):
            raise ValueError("state and observations dimensions must match the configured dimensions")
        self._validate_actions(actions, self.num_actions, (batch_size, time_steps, self.num_agents))
        if actions.device != state.device:
            raise ValueError("actions must be on the same device as state")
        return self._critic_values(state, observations, actions, self.target_critic)

    def _critic_values(self, state, observations, actions, critic):
        batch_size, time_steps = state.shape[:2]
        joint_action_features = torch.nn.functional.one_hot(
            actions.long(), num_classes=self.num_actions
        ).to(dtype=state.dtype)
        other_agent_mask = 1.0 - torch.eye(self.num_agents, device=state.device, dtype=state.dtype)
        action_features = joint_action_features.unsqueeze(2).expand(
            batch_size, time_steps, self.num_agents, self.num_agents, self.num_actions
        ) * other_agent_mask.reshape(1, 1, self.num_agents, self.num_agents, 1)
        action_features = action_features.reshape(
            batch_size, time_steps, self.num_agents, self.num_agents * self.num_actions
        )
        joint_observations = observations.reshape(
            batch_size, time_steps, self.num_agents * self.observation_dim
        ).unsqueeze(2).expand(batch_size, time_steps, self.num_agents, -1)
        agent_ids = torch.eye(self.num_agents, device=state.device, dtype=state.dtype)
        agent_ids = agent_ids.reshape(1, 1, self.num_agents, self.num_agents).expand(
            batch_size, time_steps, self.num_agents, self.num_agents
        )
        critic_inputs = torch.cat((
            state.unsqueeze(2).expand(batch_size, time_steps, self.num_agents, self.state_dim),
            joint_observations,
            action_features,
            agent_ids,
        ), dim=-1)
        return critic(critic_inputs.reshape(-1, critic_inputs.shape[-1])).reshape(
            batch_size, time_steps, self.num_agents, self.num_actions
        )

    def forward(
        self,
        state,
        observations,
        actions,
        previous_actions=None,
        initial_hidden=None,
        epsilon=None,
    ):
        if not isinstance(state, torch.Tensor) or state.ndim != 3:
            raise ValueError("state must have shape [batch, time, state_dim]")
        if not isinstance(observations, torch.Tensor) or observations.ndim != 4:
            raise ValueError("observations must have shape [batch, time, agents, observation_dim]")
        if not state.is_floating_point() or not observations.is_floating_point():
            raise ValueError("state and observations must be floating-point tensors")
        batch_size, time_steps, agents, observation_dim = observations.shape
        if (batch_size <= 0 or time_steps <= 0 or agents != self.num_agents
                or observation_dim != self.observation_dim):
            raise ValueError("observations dimensions must match the configured batch, agents, and observation_dim")
        if tuple(state.shape) != (batch_size, time_steps, self.state_dim):
            raise ValueError("state dimensions must match observations and the configured state_dim")
        if state.device != observations.device or state.dtype != observations.dtype:
            raise ValueError("state and observations must share a device and dtype")
        self._validate_actions(actions, self.num_actions, (batch_size, time_steps, self.num_agents))
        if actions.device != state.device:
            raise ValueError("actions must be on the same device as state")
        if previous_actions is None:
            previous_actions = observations.new_zeros(
                batch_size, time_steps, self.num_agents, self.num_actions
            )
        if (not isinstance(previous_actions, torch.Tensor)
                or tuple(previous_actions.shape) != (batch_size, time_steps, self.num_agents, self.num_actions)
                or not previous_actions.is_floating_point()):
            raise ValueError("previous_actions must be floating-point one-hot values matching observations")
        if previous_actions.device != state.device or previous_actions.dtype != state.dtype:
            raise ValueError("previous_actions must share state device and dtype")
        if not torch.isfinite(previous_actions).all():
            raise ValueError("previous_actions must be finite")
        is_binary = (previous_actions == 0) | (previous_actions == 1)
        row_sums = previous_actions.sum(dim=-1)
        if not torch.all(is_binary) or not torch.all((row_sums == 0) | (row_sums == 1)):
            raise ValueError("previous_actions rows must be one-hot or all zero")
        if initial_hidden is None:
            initial_hidden = observations.new_zeros(batch_size, self.num_agents, self.actor_hidden_size)
        if (not isinstance(initial_hidden, torch.Tensor)
                or tuple(initial_hidden.shape) != (batch_size, self.num_agents, self.actor_hidden_size)
                or not initial_hidden.is_floating_point()):
            raise ValueError("initial_hidden must have shape [batch, agents, actor_hidden_size]")
        if initial_hidden.device != state.device or initial_hidden.dtype != state.dtype:
            raise ValueError("initial_hidden must share state device and dtype")
        epsilon = self.epsilon_start if epsilon is None else epsilon

        agent_ids = torch.eye(self.num_agents, device=state.device, dtype=state.dtype)
        agent_ids = agent_ids.reshape(1, 1, self.num_agents, self.num_agents).expand(
            batch_size, time_steps, self.num_agents, self.num_agents
        )
        actor_inputs = torch.cat((observations, previous_actions, agent_ids), dim=-1)
        actor_inputs = torch.relu(self.actor["input"](actor_inputs))
        actor_inputs = actor_inputs.permute(0, 2, 1, 3).reshape(
            batch_size * self.num_agents, time_steps, self.actor_hidden_size
        )
        hidden = initial_hidden.reshape(batch_size * self.num_agents, self.actor_hidden_size).unsqueeze(0)
        actor_outputs, final_hidden = self.actor["gru"](actor_inputs, hidden)
        actor_outputs = actor_outputs.reshape(
            batch_size, self.num_agents, time_steps, self.actor_hidden_size
        ).permute(0, 2, 1, 3)
        logits = self.actor["output"](actor_outputs)
        probabilities = self.bounded_softmax(logits, epsilon)

        q_values = self._critic_values(state, observations, actions, self.critic)
        advantages = self.counterfactual_advantage(q_values, probabilities, actions)
        final_hidden = final_hidden.squeeze(0).reshape(batch_size, self.num_agents, self.actor_hidden_size)
        return probabilities, q_values, advantages, final_hidden
