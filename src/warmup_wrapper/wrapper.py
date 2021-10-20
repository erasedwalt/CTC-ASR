import torch


class WarmupWrapper:
    """
    Optimizer wrapper for implement warmup updates.
    Args:
        warmup (int): Number of warmup steps
        optimizer (Optimizer): Optimizer
        max_lr (float): Max lr after warmup
    """
    def __init__(self, warmup: int, optimizer: torch.optim.Optimizer, max_lr: float) -> None:
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0
        self.max_lr = max_lr
        self.warmup = warmup
        self._lrs = (torch.arange(start=0, end=warmup) / warmup) * max_lr

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if step >= self.warmup:
            return self.max_lr
        return self._lrs[step]

    def zero_grad(self):
        self.optimizer.zero_grad()
