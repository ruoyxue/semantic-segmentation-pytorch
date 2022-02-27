"""
Different Losses
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Union, List
import numpy as np


class Loss:
    def __init__(self, n_class: int, weight: Optional[torch.tensor] = None,
                 smoothing: float = 0.):
        """ base class for different losses
        :param n_class: true label is range(0, n_class)
        :param weight: weight for each class
        :param smoothing: label smoothing,
                    y(i) = smoothing / n_class, i != target
                    y(i) = 1 - smoothing + smoothing / n_class, i == target
        """
        self.n_class = n_class
        # weight
        if weight is None:
            self.weight = torch.ones(n_class, dtype=torch.float32)
        else:
            assert len(weight) == n_class, f"loss __init__ weight_dim" \
                                           f"({len(weight)}) != n_class({n_class})"
            self.weight = weight.float() * n_class / torch.sum(weight)
        # smoothing
        assert 0 <= smoothing < 1, "loss __init__ smoothing has to satisfy [0, 1), " \
                                   "got {}".format(smoothing)
        self.smoothing = smoothing
        self.loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def weighted_smoothed_one_hot(self, gts: torch.tensor):
        """ use one hot, label smoothing and weight to reconstruct gts
        :param gts: (batch_size, height, width)
        :return gts weighted one hot version, with shape (batch_size, n_class, height * width)
        """
        assert len(gts.shape) == 3, "loss weighted_one_hot gts must have 3 dimension"
        batch_size = gts.shape[0]
        off_value = self.smoothing / self.n_class
        on_value = 1. - self.smoothing + off_value
        one_hot = F.one_hot(gts.reshape(batch_size, -1), num_classes=self.n_class).transpose(2, 1)
        ont_hot = one_hot * on_value + (torch.ones_like(one_hot) - one_hot) * off_value
        return one_hot * self.weight.reshape(-1, 1)

    def __call__(self, preds: torch.tensor, gts: torch.tensor):
        raise NotImplemented

    def state_dict(self):
        raise NotImplemented

    def load_state_dict(self, state_dict: dict):
        if str(type(self)) != state_dict["criterion_type"]:
            raise TypeError("Criterion load, input dict has different criterion({}) with former "
                            "instantiation({})".format(state_dict["criterion_type"], str(type(self))))
        state_dict["weight"] = torch.from_numpy(np.array(state_dict["weight"])).float()
        state_dict["loss"] = torch.tensor([state_dict["loss"]], dtype=torch.float32)
        self.__dict__.update(state_dict)

    def to(self, device):
        """ transfer criterion to device """
        self.weight = self.weight.to(device)
        self.loss = self.loss.to(device)


class LogSoftmaxCELoss(Loss):
    """ log softmax + cross entropy loss
    """
    def __init__(self, n_class: int, weight: Optional[torch.tensor] = None,
                 smoothing: float = 0.):
        super().__init__(n_class=n_class, weight=weight, smoothing=smoothing)

    def __call__(self, preds: torch.tensor, gts: torch.tensor):
        """ calculate mean loss of the batch
        :param preds: (batch_size, n_class, height, width)
        :param gts: (batch_size, height, width)
        """
        assert preds.shape[0] == gts.shape[0], f"loss input preds has different batchsize({preds.shape[0]}) "\
                                               f"compared to that of gts({gts.shape[0]})"
        self.loss = torch.zeros_like(self.loss)
        batch_size = preds.shape[0]
        preds = F.log_softmax(preds, dim=1)
        gts = self.weighted_smoothed_one_hot(gts)
        preds = preds.reshape(batch_size, self.n_class, -1)
        # gts (batch_size, n_class, height * width)
        # preds (batch_size, n_class, height * width)
        self.loss = torch.sum(-gts * preds, dim=1)
        return torch.mean(self.loss, dim=[0, 1])

    def state_dict(self):
        return {
            "criterion_type": str(type(self)),
            "n_class": self.n_class,
            "weight": [self.weight[i].item() for i in range(self.n_class)],
            "smoothing": self.smoothing,
            "loss": self.loss.item()
        }


class SigmoidDiceLoss(Loss):
    """ sigmoid + dice loss
    dice_loss = 1 - (2 * |X ∩ Y| + eps) / (|X| + |Y| + eps)
    """
    def __init__(self, n_class: int, weight: Optional[torch.tensor] = None,
                 smoothing: float = 0., ignore_index: Union[int, List, None] = None,
                 eps: float = 1.):
        super().__init__(n_class=n_class, weight=weight, smoothing=smoothing)
        self.eps = eps
        if isinstance(ignore_index, int):
            self.ignore_index = [ignore_index]
        elif isinstance(ignore_index, list) or ignore_index is None:
            self.ignore_index = ignore_index
        else:
            raise TypeError("loss __init__ wrong type for ignore_index, which should be int or list or None")

    def __call__(self, preds: torch.tensor, gts: torch.tensor):
        """ calculate mean loss of the batch
        :param preds: (batch_size, n_class, height, width)
        :param gts: (batch_size, height, width)
        """
        assert preds.shape[0] == gts.shape[0], f"loss input preds has different batchsize({preds.shape[0]}) " \
                                               f"compared to that of gts({gts.shape[0]})"
        self.loss = torch.zeros_like(self.loss)
        batch_size = preds.shape[0]
        preds = torch.sigmoid(preds)
        gts = self.weighted_smoothed_one_hot(gts)
        # preds: (batch_size, n_class, height, width)
        # gts: (batch_size, n_class, height * width)
        count = 0
        for i in torch.arange(self.n_class):
            if self.ignore_index is None or i not in self.ignore_index:
                # take label = i as foreground, others as background
                # gts_single, preds_single: (batch_size, height * width)
                gts_single = gts[:, i]
                preds_single = preds[:, i].view(batch_size, -1)
                intersection = gts_single * preds_single
                # intersection: (batch_size, height * width)
                tem = (2 * intersection.sum(1) + self.eps) / (gts_single.sum(1) + preds_single.sum(1) + self.eps)
                self.loss += (1 - tem).mean()
                count += 1

        return self.loss / count

    def state_dict(self):
        return {
            "criterion_type": str(type(self)),
            "n_class": self.n_class,
            "weight": [self.weight[i].item() for i in range(self.n_class)],
            "smoothing": self.smoothing,
            "loss": self.loss.item(),
            "eps": self.eps,
            "ignore_index": self.ignore_index
        }


class ComposedLoss(Loss):
    """ LogSoftmaxCELoss + rate * SigmoidDiceLoss
    """
    def __init__(self, n_class: int, weight: Optional[torch.tensor] = None,
                 smoothing: float = 0., ignore_index: Union[int, List, None] = None,
                 eps: float = 1., rate: float = 1.):
        super().__init__(n_class=n_class, weight=weight, smoothing=smoothing)
        self.rate = rate
        self.eps = eps
        if isinstance(ignore_index, int):
            self.ignore_index = [ignore_index]
        elif isinstance(ignore_index, list) or ignore_index is None:
            self.ignore_index = ignore_index
        else:
            raise TypeError("loss __init__ wrong type for ignore_index, which should be int or list or None")

        self.CELoss = LogSoftmaxCELoss(n_class=self.n_class, weight=self.weight, smoothing=self.smoothing)
        self.DiceLoss = SigmoidDiceLoss(n_class=self.n_class, weight=None, smoothing=self.smoothing,
                                        ignore_index=self.ignore_index, eps=self.eps)

    def __call__(self, preds: torch.tensor, gts: torch.tensor):
        """ calculate mean loss of the batch
        :param preds: (batch_size, n_class, height, width)
        :param gts: (batch_size, height, width)
        """
        # print("CELoss", self.CELoss(preds, gts))
        # print("Dice", self.DiceLoss(preds, gts))
        return self.CELoss(preds, gts) + self.rate * self.DiceLoss(preds, gts)

    def to(self, device):
        """ transfer criterion to device """
        self.CELoss.to(device)
        self.DiceLoss.to(device)

    def state_dict(self):
        return {
            "criterion_type": str(type(self)),
            "n_class": self.n_class,
            "weight": [self.weight[i].item() for i in range(self.n_class)],
            "smoothing": self.smoothing,
            "loss": self.loss.item(),
            "eps": self.eps,
            "ignore_index": self.ignore_index,
            "rate": self.rate
        }

    def load_state_dict(self, state_dict: dict):
        if str(type(self)) != state_dict["criterion_type"]:
            raise TypeError("Criterion load, input dict has different criterion({}) with former "
                            "instantiation({})".format(state_dict["criterion_type"], str(type(self))))
        state_dict["weight"] = torch.from_numpy(np.array(state_dict["weight"])).float()
        state_dict["loss"] = torch.tensor([state_dict["loss"]], dtype=torch.float32)
        self.__dict__.update(state_dict)
        self.CELoss = LogSoftmaxCELoss(n_class=self.n_class, weight=self.weight, smoothing=self.smoothing)
        self.DiceLoss = SigmoidDiceLoss(n_class=self.n_class, weight=None, smoothing=self.smoothing,
                                        ignore_index=self.ignore_index, eps=self.eps)
