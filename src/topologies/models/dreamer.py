import math
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(input_size, hidden_size, output_size, layers):
    modules = []
    current_size = input_size
    for _ in range(layers - 1):
        modules.extend((nn.Linear(current_size, hidden_size), nn.ELU()))
        current_size = hidden_size
    modules.append(nn.Linear(current_size, output_size))
    return nn.Sequential(*modules)


class Dreamer(nn.Module):
    architecture_id = "dreamer"
    PRESETS = {
        "tiny": {
            "action_dim": 2,
            "image_size": 64,
            "image_channels": 3,
            "stoch_size": 8,
            "deter_size": 32,
            "rssm_hidden_size": 32,
            "dense_width": 32,
            "conv_depth": 4,
            "actor_layers": 2,
            "value_layers": 2,
            "reward_layers": 2,
            "imagination_horizon": 3,
            "discount": 0.99,
            "lambda_": 0.95,
            "kl_scale": 1.0,
            "free_nats": 3.0,
        },
        "small": {
            "action_dim": 6,
            "image_size": 64,
            "image_channels": 3,
            "stoch_size": 16,
            "deter_size": 128,
            "rssm_hidden_size": 128,
            "dense_width": 128,
            "conv_depth": 16,
            "actor_layers": 3,
            "value_layers": 3,
            "reward_layers": 3,
            "imagination_horizon": 10,
            "discount": 0.99,
            "lambda_": 0.95,
            "kl_scale": 1.0,
            "free_nats": 3.0,
        },
    }

    def __init__(self, action_dim=None, image_size=64, image_channels=3, stoch_size=30,
                 deter_size=200, rssm_hidden_size=200, dense_width=300, conv_depth=32,
                 actor_layers=3, value_layers=3, reward_layers=3, imagination_horizon=15,
                 discount=0.99, lambda_=0.95, kl_scale=1.0, free_nats=3.0):
        super().__init__()
        if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        dimensions = {
            "image_size": image_size,
            "image_channels": image_channels,
            "stoch_size": stoch_size,
            "deter_size": deter_size,
            "rssm_hidden_size": rssm_hidden_size,
            "dense_width": dense_width,
            "conv_depth": conv_depth,
            "actor_layers": actor_layers,
            "value_layers": value_layers,
            "reward_layers": reward_layers,
            "imagination_horizon": imagination_horizon,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if image_size != 64:
            raise ValueError("image_size must be 64")
        if image_channels != 3:
            raise ValueError("image_channels must be 3")
        for name, value in (("discount", discount), ("lambda_", lambda_),
                            ("kl_scale", kl_scale), ("free_nats", free_nats)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be between 0 and 1")
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("lambda_ must be between 0 and 1")
        if kl_scale < 0.0:
            raise ValueError("kl_scale must be nonnegative")
        if free_nats < 0.0:
            raise ValueError("free_nats must be nonnegative")
        self.action_dim = action_dim
        self.image_size = image_size
        self.image_channels = image_channels
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.rssm_hidden_size = rssm_hidden_size
        self.dense_width = dense_width
        self.conv_depth = conv_depth
        self.imagination_horizon = imagination_horizon
        self.discount = float(discount)
        self.lambda_ = float(lambda_)
        self.kl_scale = float(kl_scale)
        self.free_nats = float(free_nats)
        encoder_layers = []
        channels = image_channels
        for multiplier in (1, 2, 4, 8, 16, 32):
            next_channels = conv_depth * multiplier
            encoder_layers.extend((nn.Conv2d(channels, next_channels, 4, stride=2, padding=1), nn.ReLU()))
            channels = next_channels
        self.encoder = nn.Sequential(*encoder_layers)
        embedding_size = conv_depth * 32
        self.rssm_input = nn.Sequential(
            nn.Linear(stoch_size + action_dim, rssm_hidden_size), nn.ELU(),
            nn.Linear(rssm_hidden_size, rssm_hidden_size), nn.ELU(),
        )
        self.rssm_cell = nn.GRUCell(rssm_hidden_size, deter_size)
        self.prior_head = _mlp(deter_size, dense_width, stoch_size * 2, 2)
        self.posterior_head = _mlp(deter_size + embedding_size, dense_width, stoch_size * 2, 2)
        feature_size = deter_size + stoch_size
        self.image_decoder_input = nn.Linear(feature_size, conv_depth * 32)
        decoder_layers = []
        decoder_channels = [conv_depth * 32, conv_depth * 16, conv_depth * 8,
                            conv_depth * 4, conv_depth * 2, conv_depth, image_channels]
        for index, (input_channels, output_channels) in enumerate(pairwise(decoder_channels)):
            decoder_layers.append(nn.ConvTranspose2d(input_channels, output_channels, 4, stride=2, padding=1))
            decoder_layers.append(nn.Identity() if index == len(decoder_channels) - 2 else nn.ELU())
        self.image_decoder = nn.Sequential(*decoder_layers)
        self.reward_head = _mlp(feature_size, dense_width, 1, reward_layers)
        self.actor = _mlp(feature_size, dense_width, action_dim * 2, actor_layers)
        self.value_head = _mlp(feature_size, dense_width, 1, value_layers)

    @staticmethod
    def diagonal_kl(posterior_mean, posterior_std, prior_mean, prior_std):
        if posterior_mean.shape != posterior_std.shape or posterior_mean.shape != prior_mean.shape or posterior_mean.shape != prior_std.shape:
            raise ValueError("Gaussian means and standard deviations must have matching shapes")
        if torch.any(posterior_std <= 0) or torch.any(prior_std <= 0):
            raise ValueError("Gaussian standard deviations must be positive")
        variance_ratio = (posterior_std / prior_std).square()
        mean_term = ((posterior_mean - prior_mean) / prior_std).square()
        return (0.5 * (variance_ratio + mean_term - 1.0) + prior_std.log() - posterior_std.log()).sum(dim=-1)

    @staticmethod
    def free_nats_loss(divergence, free_nats):
        if not isinstance(free_nats, (int, float)) or isinstance(free_nats, bool) or free_nats < 0:
            raise ValueError("free_nats must be nonnegative")
        return divergence.mean().clamp_min(float(free_nats))

    @staticmethod
    def lambda_returns(rewards, values, discounts, bootstrap, lambda_):
        if rewards.shape != values.shape or rewards.shape != discounts.shape or rewards.ndim < 1:
            raise ValueError("rewards, values, and discounts must have the same nonempty shape")
        if not isinstance(lambda_, (int, float)) or isinstance(lambda_, bool) or not 0.0 <= lambda_ <= 1.0:
            raise ValueError("lambda_ must be between 0 and 1")
        bootstrap = torch.as_tensor(bootstrap, dtype=values.dtype, device=values.device)
        try:
            bootstrap = torch.broadcast_to(bootstrap, values.shape[:-1])
        except RuntimeError as error:
            raise ValueError("bootstrap must broadcast to the leading return dimensions") from error
        returns = []
        next_return = bootstrap
        for index in range(rewards.shape[-1] - 1, -1, -1):
            if index + 1 < rewards.shape[-1]:
                next_value = values[..., index + 1]
                target = (1.0 - lambda_) * next_value + lambda_ * next_return
            else:
                target = next_return
            next_return = rewards[..., index] + discounts[..., index] * target
            returns.append(next_return)
        return torch.stack(list(reversed(returns)), dim=-1)

    @staticmethod
    def unit_normal_log_prob(target, mean, event_dims=1):
        if target.shape != mean.shape:
            raise ValueError("target and mean must have matching shapes")
        if not isinstance(event_dims, int) or isinstance(event_dims, bool) or not 0 <= event_dims <= target.ndim:
            raise ValueError("event_dims must be between zero and the tensor rank")
        log_prob = -0.5 * ((target - mean).square() + math.log(2.0 * math.pi))
        if event_dims:
            log_prob = log_prob.sum(dim=tuple(range(log_prob.ndim - event_dims, log_prob.ndim)))
        return log_prob

    @staticmethod
    def tanh_gaussian_rsample(raw_mean, raw_std, epsilon=None):
        if raw_mean.shape != raw_std.shape:
            raise ValueError("raw_mean and raw_std must have matching shapes")
        mean = 5.0 * torch.tanh(raw_mean / 5.0)
        std = F.softplus(raw_std + math.log(math.expm1(5.0))) + 1e-4
        epsilon = torch.randn_like(mean) if epsilon is None else torch.as_tensor(
            epsilon, dtype=mean.dtype, device=mean.device
        )
        if epsilon.shape != mean.shape:
            raise ValueError("epsilon must match the action shape")
        return torch.tanh(mean + std * epsilon), mean, std

    def _distribution_stats(self, raw_stats):
        mean, raw_std = raw_stats.chunk(2, dim=-1)
        std = F.softplus(raw_std) + 0.1
        return mean, std

    def encode(self, observations):
        if not isinstance(observations, torch.Tensor) or observations.ndim != 5:
            raise ValueError("observations must have shape [batch, time, 3, 64, 64]")
        if observations.shape[2:] != (self.image_channels, self.image_size, self.image_size):
            raise ValueError("observations must have shape [batch, time, 3, 64, 64]")
        batch, time = observations.shape[:2]
        images = observations.reshape(batch * time, *observations.shape[2:]).float() / 255.0 - 0.5
        return self.encoder(images).flatten(start_dim=1).reshape(batch, time, -1)

    def observe(self, observations, actions):
        if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
            raise ValueError("actions must have shape [batch, time, action_dim]")
        if actions.shape[:2] != observations.shape[:2] or actions.shape[-1] != self.action_dim:
            raise ValueError("actions must match the observation batch, time, and action dimensions")
        batch, time = actions.shape[:2]
        embeddings = self.encode(observations)
        previous_actions = torch.cat((actions.new_zeros(batch, 1, self.action_dim), actions[:, :-1]), dim=1)
        belief = embeddings.new_zeros(batch, self.deter_size)
        latent = embeddings.new_zeros(batch, self.stoch_size)
        beliefs = []
        prior_means = []
        prior_stds = []
        prior_samples = []
        posterior_means = []
        posterior_stds = []
        posterior_samples = []
        for index in range(time):
            recurrent_input = self.rssm_input(torch.cat((latent, previous_actions[:, index]), dim=-1))
            belief = self.rssm_cell(recurrent_input, belief)
            prior_mean, prior_std = self._distribution_stats(self.prior_head(belief))
            posterior_mean, posterior_std = self._distribution_stats(
                self.posterior_head(torch.cat((belief, embeddings[:, index]), dim=-1))
            )
            latent = posterior_mean + posterior_std * torch.randn_like(posterior_mean)
            beliefs.append(belief)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            prior_samples.append(prior_mean + prior_std * torch.randn_like(prior_mean))
            posterior_means.append(posterior_mean)
            posterior_stds.append(posterior_std)
            posterior_samples.append(latent)
        belief_sequence = torch.stack(beliefs, dim=1)
        prior_mean_sequence = torch.stack(prior_means, dim=1)
        prior_std_sequence = torch.stack(prior_stds, dim=1)
        prior_sample_sequence = torch.stack(prior_samples, dim=1)
        posterior_mean_sequence = torch.stack(posterior_means, dim=1)
        posterior_std_sequence = torch.stack(posterior_stds, dim=1)
        posterior_sample_sequence = torch.stack(posterior_samples, dim=1)
        features = torch.cat((posterior_sample_sequence, belief_sequence), dim=-1)
        return {
            "embeddings": embeddings,
            "belief": belief_sequence,
            "prior_mean": prior_mean_sequence,
            "prior_std": prior_std_sequence,
            "prior_sample": prior_sample_sequence,
            "posterior_mean": posterior_mean_sequence,
            "posterior_std": posterior_std_sequence,
            "posterior_sample": posterior_sample_sequence,
            "features": features,
        }

    def decode_images(self, features):
        batch, time = features.shape[:2]
        hidden = self.image_decoder_input(features.reshape(batch * time, -1))
        hidden = hidden.reshape(batch * time, self.conv_depth * 32, 1, 1)
        return self.image_decoder(hidden).reshape(batch, time, self.image_channels, self.image_size, self.image_size)

    def actor_parameters(self, features):
        raw_mean, raw_std = self.actor(features).chunk(2, dim=-1)
        return raw_mean, raw_std

    def deterministic_action(self, features):
        raw_mean, _ = self.actor_parameters(features)
        return torch.tanh(5.0 * torch.tanh(raw_mean / 5.0))

    def _imagine(self, initial_features):
        sample = initial_features[..., :self.stoch_size].reshape(-1, self.stoch_size)
        belief = initial_features[..., self.stoch_size:].reshape(-1, self.deter_size)
        imagined_features = []
        imagined_actions = []
        imagined_rewards = []
        imagined_values = []
        for _ in range(self.imagination_horizon):
            feature = torch.cat((sample, belief), dim=-1)
            raw_mean, raw_std = self.actor_parameters(feature)
            action, _, _ = self.tanh_gaussian_rsample(raw_mean, raw_std)
            recurrent_input = self.rssm_input(torch.cat((sample, action), dim=-1))
            belief = self.rssm_cell(recurrent_input, belief)
            prior_mean, prior_std = self._distribution_stats(self.prior_head(belief))
            sample = prior_mean + prior_std * torch.randn_like(prior_mean)
            next_feature = torch.cat((sample, belief), dim=-1)
            imagined_features.append(next_feature)
            imagined_actions.append(action)
            imagined_rewards.append(self.reward_head(next_feature).squeeze(-1))
            imagined_values.append(self.value_head(next_feature).squeeze(-1))
        features = torch.stack(imagined_features, dim=1)
        actions = torch.stack(imagined_actions, dim=1)
        rewards = torch.stack(imagined_rewards, dim=1)
        values = torch.stack(imagined_values, dim=1)
        discounts = torch.full_like(rewards, self.discount)
        bootstrap = values[..., -1]
        returns = self.lambda_returns(
            rewards[..., :-1], values[..., :-1], discounts[..., :-1], bootstrap, self.lambda_
        )
        prior_discounts = torch.cat((torch.ones_like(discounts[..., :1]), discounts[..., :-1]), dim=-1)
        weights = torch.cumprod(prior_discounts, dim=-1)
        actor_loss = -(weights[..., :-1] * returns).mean()
        value_loss = (weights[..., :-1] * 0.5 * (values[..., :-1] - returns.detach()).square()).mean()
        return {
            "imagined_features": features,
            "imagined_actions": actions,
            "imagined_rewards": rewards,
            "imagined_values": values,
            "imagined_returns": returns,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
        }

    def forward(self, observations, actions, rewards=None):
        output = self.observe(observations, actions)
        if rewards is None:
            rewards = actions.new_zeros(actions.shape[:2])
        if not isinstance(rewards, torch.Tensor) or rewards.shape != actions.shape[:2]:
            raise ValueError("rewards must have shape [batch, time]")
        reconstruction = self.decode_images(output["features"])
        reward = self.reward_head(output["features"]).squeeze(-1)
        image_log_prob = self.unit_normal_log_prob(
            observations.float() / 255.0 - 0.5, reconstruction, event_dims=3
        )
        reward_log_prob = self.unit_normal_log_prob(
            rewards.to(dtype=reward.dtype, device=reward.device), reward, event_dims=0
        )
        kl = self.diagonal_kl(
            output["posterior_mean"], output["posterior_std"],
            output["prior_mean"], output["prior_std"],
        )
        world_loss = -image_log_prob.mean() - reward_log_prob.mean() + self.kl_scale * self.free_nats_loss(
            kl, self.free_nats
        )
        output.update({"reconstruction": reconstruction, "reward": reward, "kl": kl,
                       "world_loss": world_loss})
        output.update(self._imagine(output["features"]))
        return output
