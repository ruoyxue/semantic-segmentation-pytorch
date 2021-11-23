import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, cohen_kappa_score
import logging
from typing import List


class Evaluator:
    def __init__(self):
        self.count = 0  # count times of accumulation
        self.metrics: dict = {}  # save metrics

    def accumulate(self, preds: torch.Tensor, labels: torch.Tensor) -> dict:
        """ Accumulate metrics """
        raise NotImplementedError

    def clear(self):
        """ Clear metrics and count """
        self.count = 0
        self.metrics.clear()

    def log_metrics(self):
        """ Print metrics using logging.info """
        raise NotImplementedError


class ClassificationEvaluator(Evaluator):
    """ Evaluator for classification task """
    def __init__(self, true_label):
        super().__init__()
        self.true_label = true_label
        self.labels = []
        self.preds = []

    @torch.no_grad()
    def accumulate(self, preds: torch.Tensor, labels: torch.Tensor):
        assert preds.shape == labels.shape, \
            f"preds {preds.shape} has different shape compared with labels {labels.shape}"
        self.preds += list(preds.detach().cpu().numpy())
        self.labels += list(labels.detach().cpu().numpy())

    def log_metrics(self):
        args = {
            "y_pred": np.array(self.preds),
            "y_true": np.array(self.labels),
            "average": "macro",
            "labels": self.true_label,
            "zero_division": 0
        }
        precision = round(precision_score(**args), 4)
        recall = round(recall_score(**args), 4)
        f_score = round(f1_score(**args), 4)
        logging.info(f"Precision: {precision}    Recall: {recall}    F1_score: {f_score}")


class SegmentationEvaluator(Evaluator):
    """ Evaluator for segmentation task
    :param true_label: list of true labels, e.g. [0, 1, 2, 3, 4]
    """
    def __init__(self, true_label):
        super().__init__()
        self.true_label = true_label

    def accumulate(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        :param preds: predictions (batch_size, height, width)
        :param labels: labels (batch_size, height, width)
        """
        assert preds.shape == labels.shape,\
            "evaluator preds.shape != labels.shape"
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        n = preds.shape[0]
        for i in range(n):
            self.count += 1
            if self.count == 1:
                self.metrics["miou"] = self.mean_iou(preds[i, :, :], labels[i, :, :])
                self.metrics["kappa"] = self.kappa(preds[i, :, :], labels[i, :, :])
            else:
                self.metrics["miou"] += self.mean_iou(preds[i, :, :], labels[i, :, :])
                self.metrics["kappa"] += self.kappa(preds[i, :, :], labels[i, :, :])

    def log_metrics(self):
        miou = round(self.metrics["miou"] / self.count, 4)
        kappa = round(self.metrics["kappa"] / self.count, 4)
        logging.info(f"miou: {miou}    kappa: {kappa}")

    @staticmethod
    def iou(pred, label, pos_label: int) -> float:
        """ compute iou for binary problems
        :param pred: 2d tensor, prediction
        :param label: 2d tensor, label
        :param pos_label: take pos_label as positive label, others as negative
        :return miou: float, iou
        """
        intersection = np.logical_and(pred == pos_label, label == pos_label)
        union = np.logical_or(pred == pos_label, label == pos_label)
        return np.sum(intersection) / (np.sum(union) + 1)

    def mean_iou(self, pred, label):
        """ compute mean iou
        :param pred: 2d tensor, prediction
        :param label: 2d tensor, label
        :return miou: float, mean iou
        """
        miou = 0
        for i in self.true_label:
            miou += self.iou(pred, label, pos_label=i)
        return miou / len(self.true_label)

    def kappa(self, pred, label):
        """ compute kappa coefficient
        :param pred: 2d tensor, prediction
        :param label: 2d tensor, label
        :return miou: float, kappa
        """
        pred = np.array(pred).flatten()
        label = np.array(label).flatten()
        return cohen_kappa_score(label, pred, labels=self.true_label)


