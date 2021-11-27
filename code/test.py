"""
function for testing

TODO: add torch.jit.script
"""



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


@torch.no_grad()
def tester(test_args: argparse):
    logging.info(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
    # 1. -------------Prepare testloader, evaluator---------------------------
    test_dataloader = PNGTestloader(image_path=os.path.join(test_args.test_data_path, "image"),
                                    chip_size=test_args.chip_size, stride=test_args.stride,
                                    n_class=test_args.n_class, batch_size=test_args.batch_size,
                                    device=test_args.device)

    evaluator = SegmentationEvaluator(true_label=range(test_args.n_class))

    # 2. ----------------------------Testing Process and evaluation-------------------------------------
    test_args.model.load_state_dict(torch.load(test_args.load_model_path))
    test_args.model.eval()
    test_args.model.to(test_args.device)
    max_batch_num = np.ceil(len(test_dataloader) / test_args.batch_size)
    with tqdm(total=max_batch_num, unit_scale=True, unit=" img", colour="cyan", ncols=60) as pbar:
        for i, (data, info) in enumerate(test_dataloader):
            data = data.to(test_args.device)  # data (batch_size, channels, height, width)
            preds = test_args.model(data)  # preds (batch_size, n_class, height, width)
            for whole_label, image_name in test_dataloader.stitcher(preds, info, i == (max_batch_num-1)):
                # before: whole label (n_class, height, width)
                whole_label = torch.argmax(whole_label, dim=0)
                # after: whole label (height, width)
                gt = torch.tensor(cv2.imread(os.path.join(test_args.test_data_path, "gt", image_name))[:, :, 0])
                evaluator.accumulate(whole_label, gt.to(test_args.device))
                cv2.imwrite(os.path.join(test_args.save_output_path, image_name), whole_label.cpu().numpy())
            pbar.update(1)
    evaluator.log_metrics()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = TestArgs()
    tester(args.args)

