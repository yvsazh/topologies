"""PPO recipe (Level 3 fit-loop with dummy environment, no Gym dependency)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from topologies.training.recipe_base import (
    EpochContext,
    EpochResult,
    FitContext,
    FitResult,
    Recipe,
    RecipeComponents,
    RecipeConfig,
    StepContext,
    StepResult,
    register_recipe,
)


class PPOComponents(RecipeComponents):
    env: Any = None
    reward_fn: Callable | None = None


class PPOConfig(RecipeConfig):
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    rollout_steps: int = 32
    epochs_per_rollout: int = 2
    obs_dim: int = 8
    act_dim: int = 4


@dataclass
class DummyEnv:
    """Minimal discrete environment for tests (no Gym dependency)."""

    obs_dim: int = 8
    act_dim: int = 4
    max_steps: int = 16
    _t: int = 0

    def reset(self) -> torch.Tensor:
        self._t = 0
        return torch.randn(self.obs_dim)

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict]:
        self._t += 1
        obs = torch.randn(self.obs_dim)
        reward = float(torch.randn(()).item())
        done = self._t >= self.max_steps
        return obs, reward, done, {}


class PPOPolicyHead(nn.Module):
    """Tiny actor-critic head used when the model only produces logits."""

    def __init__(self, obs_dim: int = 8, act_dim: int = 4) -> None:
        super().__init__()
        self.actor = nn.Linear(obs_dim, act_dim)
        self.critic = nn.Linear(obs_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)


@register_recipe("ppo", help="PPO with rollout collection (dummy env for tests).")
class PPORecipe(Recipe):
    level = "fit"
    ConfigSchema = PPOConfig
    ComponentsSchema = PPOComponents

    def default_training_policy(self) -> dict:
        return {"optimizer": "adam", "lr": 3e-4, "weight_decay": 0.0, "scheduler": None}

    def expected_batch_spec(self) -> dict[str, str]:
        # Level 3: no ordinary DataLoader requirement; batches are rollouts.
        return {"obs": "FloatTensor[N, obs_dim] (rollout, not DataLoader)"}

    def dummy_batch(self, device: torch.device | str = "cpu") -> dict:
        dev = torch.device(str(device))
        torch.manual_seed(0)
        return {"obs": torch.randn(8, 8, device=dev)}

    def _policy_forward(
        self, model: nn.Module, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = model(obs)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[0], out[1].reshape(-1)
        if isinstance(out, dict) and "logits" in out and "value" in out:
            return out["logits"], out["value"].reshape(-1)
        raise TypeError("PPO model must return (policy_logits, value) or {'logits': ..., 'value': ...}")

    def collect_rollout(
        self, model: nn.Module, env: Any, cfg: PPOConfig, device: torch.device
    ) -> dict[str, torch.Tensor]:
        model.eval()
        obs_list, act_list, rew_list, logp_list, val_list, bootstrap_list = [], [], [], [], [], []
        terminated_list, truncated_list = [], []
        obs = env.reset()
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        for _ in range(cfg.rollout_steps):
            o = obs.to(device).float().unsqueeze(0)
            with torch.no_grad():
                logits, value = self._policy_forward(model, o)
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
            action = int(act.item())
            transition = env.step(action)
            if len(transition) == 5:
                obs, reward, terminated, truncated, _ = transition
            else:
                obs, reward, done, _ = transition
                terminated, truncated = bool(done), False
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32)
            obs_list.append(o.squeeze(0).cpu())
            act_list.append(act.cpu())
            rew_list.append(float(reward))
            logp_list.append(logp.cpu())
            val_list.append(value.cpu())
            with torch.no_grad():
                _, bootstrap_value = self._policy_forward(model, obs.to(device).float().unsqueeze(0))
            bootstrap_list.append(bootstrap_value.squeeze(0).cpu())
            terminated_list.append(bool(terminated))
            truncated_list.append(bool(truncated))
            episode_end = bool(terminated or truncated)
            # A terminal transition has no bootstrap; time-limit truncation does.
            if terminated:
                bootstrap_list[-1] = torch.zeros_like(bootstrap_list[-1])
            if episode_end:
                obs = env.reset()
                if not isinstance(obs, torch.Tensor):
                    obs = torch.as_tensor(obs, dtype=torch.float32)
        model.train()
        obs_t = torch.stack(obs_list)
        acts = torch.cat(act_list).long()
        rews = torch.tensor(list(rew_list), dtype=torch.float32)
        old_logp = torch.cat(logp_list)
        vals = torch.cat(val_list)
        # Simple discounted returns + GAE-lite advantages.
        returns = torch.zeros_like(rews)
        running = 0.0
        for i in reversed(range(len(rews))):
            # Each stored bootstrap is the value of the actual successor state;
            # prevent advantage/return recursion from crossing an episode boundary.
            next_return = bootstrap_list[i] if (i == len(rews) - 1 or truncated_list[i]) else running
            if terminated_list[i]:
                next_return = torch.zeros_like(next_return)
            elif i < len(rews) - 1 and (terminated_list[i] or truncated_list[i]):
                running = torch.zeros_like(running)
            running = rews[i] + cfg.gamma * next_return
            returns[i] = running
        adv = returns - vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return {
            "obs": obs_t, "actions": acts, "returns": returns,
            "advantages": adv, "old_logp": old_logp, "values": vals,
        }

    def ppo_loss(
        self, model: nn.Module, rollout: dict, cfg: PPOConfig, device: torch.device
    ) -> tuple[torch.Tensor, dict[str, float]]:
        obs = rollout["obs"].to(device).float()
        logits, values = self._policy_forward(model, obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(rollout["actions"].to(device))
        entropy = dist.entropy().mean()
        ratio = torch.exp(logp - rollout["old_logp"].to(device))
        adv = rollout["advantages"].to(device)
        pg1 = ratio * adv
        pg2 = torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv
        pg_loss = -torch.min(pg1, pg2).mean()
        v_loss = F.mse_loss(values, rollout["returns"].to(device))
        loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy
        return loss, {
            "loss": float(loss.detach().cpu()),
            "pg": float(pg_loss.detach().cpu()),
            "v": float(v_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
        }

    # Level-2 style single update (used by Trainer epoch path if wired manually).
    def training_step(self, ctx: StepContext) -> StepResult:
        raise NotImplementedError("PPO uses fit_loop(); provide env via components.")

    def epoch_loop(self, ctx: EpochContext) -> EpochResult:
        raise NotImplementedError("PPO uses fit_loop().")

    def fit_loop(self, ctx: FitContext) -> FitResult:
        trainer = ctx.trainer
        cfg: PPOConfig = ctx.recipe_config
        comp: PPOComponents = ctx.components
        env = (comp.env if comp is not None else None) or DummyEnv(
            obs_dim=cfg.obs_dim, act_dim=cfg.act_dim
        )
        history: list[dict] = []
        # Use Trainer-provided services: device, logging, checkpoint, stop checks.
        total_epochs = trainer.config.max_epochs if trainer is not None else 1
        for epoch in range(total_epochs):
            rollout = self.collect_rollout(ctx.model, env, cfg, ctx.device)
            for _ in range(cfg.epochs_per_rollout):
                loss, metrics = self.ppo_loss(ctx.model, rollout, cfg, ctx.device)
                trainer._opt_zero_grad("default") if trainer else None
                trainer._opt_backward(loss, "default") if trainer else loss.backward()
                if trainer:
                    trainer._opt_clip("default")
                    trainer._opt_step("default")
                    trainer._sched_step("default")
            if trainer is not None:
                trainer._log(epoch, metrics)
                history.append({"epoch": epoch, **metrics})
                trainer.history.append({"epoch": epoch, **metrics})
                if trainer._should_stop(total_epochs, epoch):
                    break
        return FitResult(
            global_step=trainer.global_step if trainer else 0,
            epochs=len(history),
            history=history,
        )
