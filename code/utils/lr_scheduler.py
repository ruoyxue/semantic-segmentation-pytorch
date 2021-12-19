"""
Different lr schedulers
"""
import logging

import torch
import copy


class PlateauLRScheduler:
    """Analogue to torch.optim.lr_scheduler.ReduceLROnPlateau
    We use list to record initial_lr for future upgrade
    warm up reference: https://blog.csdn.net/qq_35091353/article/details/117322293
    TODO: Add programs to schedule several optimizers synchronously
    TODO: replace many if clauses with better algorithm

    :param optimizer: optimizer to be scheduled
    :param lr_factor: every lr upgrade would be lr = lr * lr_factor
    :param mode: "min" means metric smaller is better, "max" is the opposite
    :param patience: we can endure 'patience' times of bad metrics when step is called
    :param min_lr: min lr we can endure
    :param threshold: "min" mode, metric descends smaller than lr * (1-threshold) will be
                      regarded as bad epoch, while "max" mode is lr * (1 - threshold)
    :param warmup_duration: epochs for warm up
    """
    def __init__(self, optimizer, lr_factor: float, mode: str = "min", patience: int = 0,
                 min_lr: float = 1e-8, threshold: float = 1e-5, warmup_duration: int = 0):

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"Scheduler expects torch optimizer, got {type(optimizer)}")
        if mode not in ["min", "max"]:
            raise ValueError(f"Scheduler expects mode in [\"min\", \"max\"], got {mode}")
        assert 0 < lr_factor < 1, f"Scheduler expects factor falls in (0, 1), got {lr_factor}"
        assert warmup_duration >= 0, f"Scheduler expects warmup_duration >= 0, got {warmup_duration}"

        self.optimizer = optimizer
        self.lr_factor = lr_factor
        self.mode = mode
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.warmup_duration = warmup_duration
        self.initial_lr = optimizer.param_groups[0]["lr"]
        assert 0 < min_lr < self.initial_lr, f"Scheduler expects min_lr less than lr in optimizer, " \
                                             f"got min_lr: {self.min_lr}, optimizer lr: {self.initial_lr}"

        self._warmup_count = 1  # epochs that warm up has gone through
        self._bad_count = 0
        self.best_metric = 0
        self.current_lr = self.initial_lr

        if self.warmup_duration > 0:
            self.current_lr = self.initial_lr / self.warmup_duration
            self.optimizer.param_groups[0]["lr"] = self.current_lr

    def step(self, current_metric, epoch: int):
        """ analogous to step function in torn.optim.lr_scheduler """
        # warm up step
        if epoch == 1:
            self.best_metric = copy.deepcopy(current_metric)

        if self._warmup_count < self.warmup_duration:
            self.current_lr += self.initial_lr / self.warmup_duration
            self.optimizer.param_groups[0]["lr"] = self.current_lr
            if self._is_better(current_metric):
                self.best_metric = current_metric
            self._warmup_count += 1
            logging.info(f"epoch {epoch}: warm up adjust lr to {self.current_lr}")
        else:
            # common step
            if self._is_better(current_metric):
                self.best_metric = current_metric
                self._bad_count = 0
            else:
                self._bad_count += 1
                if self._bad_count > self.patience and self.current_lr > self.min_lr:
                    self.current_lr = max(self.current_lr * self.lr_factor, self.min_lr)
                    self.optimizer.param_groups[0]["lr"] = self.current_lr
                    logging.info(f"epoch {epoch}: reduce lr to {self.current_lr}")
                    self._bad_count = 0

    def _is_better(self, current_metric):
        """ test if the current metric is better than the best """
        if self.mode == "min":
            return current_metric < self.best_metric * (1 - self.threshold)
        else:
            return current_metric > self.best_metric * (1 + self.threshold)

    def state_dict(self):
        """ add type check to former state dict in torch """
        return {
            "scheduler_type": str(type(self)),
            "optimizer_type": str(type(self.optimizer)),
            "lr_factor": self.lr_factor,
            "mode": self.mode,
            "patience": self.patience,
            "min_lr": self.min_lr,
            "threshold": self.threshold,
            "warmup_duration": self.warmup_duration,
            "_warmup_count": self._warmup_count,
            "_bad_count": self._bad_count,
            "best_metric": self.best_metric,
            "current_lr": self.current_lr,
            "initial_lr": self.initial_lr
        }

    def load_state_dict(self, state_dict: dict):
        if str(type(self)) != state_dict["scheduler_type"]:
            raise TypeError("Scheduler load, input dict has different scheduler_type({}) with former "
                            "instantiation({})".format(state_dict["scheduler_type"], str(type(self))))
        if str(type(self.optimizer)) != state_dict["optimizer_type"]:
            raise TypeError("Scheduler load, input dict has different optimizer_type({}) with former "
                            "instantiation({})".format(state_dict["optimizer_type"], str(type(self.optimizer))))
        assert self.initial_lr == state_dict["initial_lr"],\
            "Scheduler load, optimizer has different initial_lr({}) with that in input dict({})"\
            .format(self.initial_lr, state_dict["initial_lr"])
        self.__dict__.update(state_dict)

    def get_lr(self):
        return round(self.current_lr, 5)