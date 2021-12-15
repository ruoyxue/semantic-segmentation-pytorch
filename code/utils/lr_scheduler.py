"""
Different lr schedulers
"""

import torch


class PlateauLRScheduler:
    """Analogue to torch.optim.lr_scheduler.ReduceLROnPlateau
    We use list to record initial_lr for future upgrade
    warm up reference: https://blog.csdn.net/qq_35091353/article/details/117322293
    TODO: Add programs to schedule several optimizers synchronously

    :param optimizer: optimizer to be scheduled
    :param lr_factor: every lr upgrade would be lr = lr * lr_factor
    :param mode: "min" means metric smaller is better, "max" is the opposite
    :param patience: we can endure 'patience' times of bad metrics when step is called
    :param min_lr: min lr we can endure
    :param threshold: "min" mode, metric descends smaller than lr * (1-threshold) will be
                      regarded as bad epoch, while "max" mode is lr * (1 - threshold)
    :param verbose: whether to show current_lr when lr is upgraded
    :param cool_down: every time lr is upgraded, we cannot record bad epochs until cool_down period
                      has been went through
    :param warmup_duration: epochs for warm up
    :param warmup_factor: warm up start with this lr * warmup_factor, finally reach lr in optimizer
    """
    def __init__(self, optimizer, lr_factor: float, mode: str = "min", patience: int = 0,
                 min_lr: float = 1e-8, threshold: float = 1e-6, verbose: str = False,
                 cool_down: int = 0, warmup_duration: int = 0, warmup_factor: float = 1):

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"Scheduler expects torch optimizer, got {type(optimizer)}")
        if mode not in ["min", "max"]:
            raise ValueError(f"Scheduler expects mode in [\"min\", \"max\"], got {mode}")
        assert 0 < lr_factor < 1, f"Scheduler expects factor falls in (0, 1), got {lr_factor}"
        assert cool_down >= 0, f"Scheduler expects cool_down >= 0, got {cool_down}"
        assert warmup_duration >= 0, f"Scheduler expects warmup_duration >= 0, got {warmup_duration}"
        assert 0 < warmup_factor <= 1, f"Scheduler expects warmup_factor falls in (0,1], got {warmup_factor}"

        self.initial_lr = [optimizer.param_groups[0]["lr"]]
        assert 0 < min_lr < self.initial_lr[0], f"Scheduler expects min_lr less than initial_lr in optimizer, " \
                                                f"got min_lr: {min_lr}, optimizer lr: {self.initial_lr[0]}"
        self.optimizer = optimizer
        self.lr_factor = lr_factor
        self.mode = mode
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.verbose = verbose
        self.cool_down = cool_down
        self.warmup_duration = warmup_duration
        self.warmup_factor = warmup_factor
        self.current_lr = 0
        self.current_epoch = 0
        self._step_count = 0
        self._cool_down_count = 0

        