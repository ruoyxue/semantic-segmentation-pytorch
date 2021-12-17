"""
Different Losses
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional


class LogSoftmaxCELoss:
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
            self.weight = torch.ones(n_class, dtype=torch.float32)
        else:
            assert len(weight) == n_class, f"loss __init__ weight_dim" \
                                           f"({len(weight)}) != n_class({n_class})"
            self.weight = weight.float() * n_class / torch.sum(weight)
        # smoothing
        assert 0 <= smoothing < 1, "loss __init__ smoothing has to satisfy [0, 1), " \
                                   "got {}".format(smoothing)
        self.off_value = smoothing / n_class
        self.on_value = 1. - smoothing + self.off_value
        self.loss = torch.zeros(1, dtype=torch.float32, requires_grad=True)

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
        for i in torch.arange(batch_size):
            gt = self.one_hot(gts[i]) * self.weight.reshape(-1, 1)  # use broadcasting
            # print(gt)
            pred = preds[i].reshape((self.n_class, -1))
            '''if torch.isnan(torch.sum(gt)):
                print("gt is nan")
            if torch.isnan(torch.sum(pred)):
                print("pred is nan")
            if torch.isnan(torch.sum(-gt * pred)):
                print("sum is nan")
                print("sum", torch.sum(-gt * pred))
                print("gt", gt)
                print("pred", pred)
                break'''
            self.loss += torch.sum(-gt * pred) / pred.shape[1]
        '''if torch.isnan(self.loss):
            print("loss is nan")
            print("loss", self.loss)'''
        return self.loss / batch_size

    def one_hot(self, gt: torch.tensor):
        """ one hot of gt
        :param gt: (height, width)
        :return gt one hot version, with shape (n_class, height * width)
        """
        # one_hot (n_class, height * width)
        one_hot = F.one_hot(gt.reshape(-1), num_classes=self.n_class).T
        return one_hot * self.on_value + (torch.ones_like(one_hot) - one_hot) * self.off_value

    def to(self, device):
        """ transfer weight to device """
        self.weight = self.weight.to(device)
        self.loss = self.loss.to(device)

    def state_dict(self):
        return {
            "criterion_type": type(self),
            "n_class": self.n_class,
            "weight": self.weight,
            "off_value": self.off_value,
            "on_value": self.on_value,
            "loss": self.loss
        }

    def load_state_dict(self, state_dict: dict):
        if type(self) is not state_dict["criterion_type"]:
            raise TypeError("Criterion load, input dict has different criterion({}) with former "
                            "instantiation({})".format(state_dict["criterion_type"], type(self)))
        self.__dict__.update(state_dict)
