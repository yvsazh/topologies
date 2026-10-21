"""Lookahead wrapper around a PyTorch inner optimizer (Adam by default)."""

import torch


def lookahead(params, *, lr, weight_decay=0.0, **backend_options):
    """Create a Lookahead optimizer using ``k`` fast steps and slow interpolation."""
    k = backend_options.pop("k", 5)
    alpha = backend_options.pop("alpha", 0.5)
    base_optimizer = backend_options.pop("base_optimizer", torch.optim.Adam)
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    class _Lookahead(torch.optim.Optimizer):
        def __init__(self, parameters):
            self.optimizer = base_optimizer(
                parameters, lr=lr, weight_decay=weight_decay, **backend_options
            )
            self.param_groups = self.optimizer.param_groups
            self.defaults = self.optimizer.defaults
            self.state = self.optimizer.state
            self._steps = 0
            self._slow = [[p.detach().clone() for p in group["params"]]
                          for group in self.param_groups]

        @torch.no_grad()
        def step(self, closure=None):
            loss = self.optimizer.step(closure)
            self._steps += 1
            if self._steps % k == 0:
                for group, slow_group in zip(self.param_groups, self._slow):
                    for parameter, slow in zip(group["params"], slow_group):
                        slow.add_(parameter - slow, alpha=alpha)
                        parameter.copy_(slow)
            return loss

        def zero_grad(self, set_to_none=True):
            self.optimizer.zero_grad(set_to_none=set_to_none)

        def state_dict(self):
            return {"optimizer": self.optimizer.state_dict(), "slow": self._slow,
                    "steps": self._steps}

        def load_state_dict(self, state_dict):
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self._slow = state_dict["slow"]
            self._steps = state_dict["steps"]
            self.param_groups = self.optimizer.param_groups
            self.state = self.optimizer.state

    return _Lookahead(params)
