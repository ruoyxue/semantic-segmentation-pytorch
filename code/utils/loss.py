"""
Different Losses

"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional


class LogSoftmaxCrossEntropyLoss:
    """ log softmax + cross entropy loss
    :param n_class: true label is range(0, n_class)
    :param weight: weight for each class
    :param smoothing: label smoothing,
                    y(i) = smoothing / n_class, i != target
                    y(i) = 1 - smoothing + smoothing / n_class, i == target
    """
    def __init__(self, n_class: int, weight: Optional[torch.tensor] = None,
                 smoothing: float = 0.):
        self.n_class = n_class
        # weight
        if weight is None:
            self.weight = torch.ones(n_class)
        else:
            assert len(weight) == n_class, f"loss __init__ weight dim" \
                                           f"({len(weight)}) != n_class({n_class})"
            self.weight = weight
        # smoothing
        assert 0 <= smoothing < 1, "loss __init__ smoothing has to satisfy [0, 1), " \
                                   "got {}".format(smoothing)
        self.off_value = smoothing / n_class
        self.on_value = 1. - smoothing + self.off_value

    def __call__(self, preds: torch.tensor, gts: torch.tensor):
        """ return sum loss of the batch
        :param preds: (batch_size, n_class, height, width)
        :param gts: (batch_size, height, width)
        """
        assert preds.shape[0] == gts.shape[0], f"loss input preds has different batchsize({preds.shape[0]}) "\
                                               f"compared to that of gts({gts.shape[0]})"
        batch_size = preds.shape[0]
        loss = torch.Tensor([0])
        preds = F.log_softmax(preds, dim=1)
        for i in torch.arange(batch_size):
            gt = self.one_hot(gts[i]) * self.weight.T   # use broadcasting
            pred = preds[i].reshape((self.n_class, -1))
            loss -= gt * pred / torch.sum(self.weight)  # reduce floating point underflow
        return loss

    def one_hot(self, gt: torch.tensor):
        """ one hot of gt
        :param gt: (height, width)
        :return gt one hot version, with shape (n_class, height * width)
        """
        # one_hot (n_class, height * width)
        one_hot = F.one_hot(gt.reshape(-1), num_classes=self.n_class).T
        return one_hot * self.on_value + (not one_hot) * self.off_value


