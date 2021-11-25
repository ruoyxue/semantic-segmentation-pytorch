import logging
import datetime
import os
import numpy as np
from tqdm import tqdm
import torch
from utils import PNGTestloader
import argparse
from utils.evaluator import SegmentationEvaluator
from torchvision import datasets, transforms
from args import TestArgs
import cv2


def tester(test_args: argparse):
    logging.info(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
    # 1. -------------Prepare testloader, evaluator---------------------------
    test_dataloader = PNGTestloader(image_path=os.path.join(test_args.test_data_path, "image"),
                                    chip_size=test_args.chip_size, stride=test_args.stride,
                                    n_class=test_args.n_class, batch_size=test_args.batch_size)

    evaluator = SegmentationEvaluator(true_label=range(test_args.n_class))

    # 2. ----------------------------Testing Process and evaluation-------------------------------------
    test_args.model.load_state_dict(torch.load(test_args.load_model_path))
    test_args.model.eval()
    test_args.model.to(test_args.device)
    with tqdm(total=np.ceil(len(test_dataloader) / test_args.batch_size),
              unit_scale=True, unit=" img", colour="cyan", ncols=60) as pbar:
        for i, (data, info) in enumerate(test_dataloader):
            data = data.to(test_args.device)
            # labels = labels.to(test_args.device)
            preds = test_args.model(data)
            for whole_label, image_name in test_dataloader.stitcher(preds, info):
                whole_label = np.array(whole_label).argmax(axis=0)
                cv2.imwrite(os.path.join(test_args.save_output_path, image_name), whole_label)
            # evaluator.accumulate(torch.argmax(preds, dim=1), labels)
            pbar.update(1)

    # logging.info(evaluator.count)
    # evaluator.log_metrics()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = TestArgs()
    tester(args.args)

