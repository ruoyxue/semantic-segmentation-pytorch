import logging
import datetime

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from utils.evaluator import SegmentationEvaluator
from torchvision import datasets, transforms
from args import TestArgs


def tester(test_args: argparse):
    logging.info(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
    # 1. -------------Prepare dataset, dataloader, evaluator---------------------------
    test_dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../dataset/fashionmnist_data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_args.batch_size)
    evaluator = SegmentationEvaluator(true_label=range(test_args.n_class))

    # 2. ----------------------------Testing Process and evaluation-------------------------------------
    test_args.model.load_state_dict(torch.load(test_args.load_model_path))
    test_args.model.eval()
    test_args.model.to(test_args.device)
    with tqdm(total=np.ceil(len(test_dataloader) / test_args.batch_size),
              unit_scale=True, unit=" img", colour="cyan", ncols=60) as pbar:
        for batch_num, (images, labels) in enumerate(test_dataloader):
            images = images.to(test_args.device)
            labels = labels.to(test_args.device)
            preds = test_args.model(images)

            evaluator.accumulate(torch.argmax(preds, dim=1), labels)
            pbar.update(1)

    logging.info(evaluator.count)
    evaluator.log_metrics()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = TestArgs()
    tester(args.args)

