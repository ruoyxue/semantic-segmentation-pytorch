"""
function for testing

TODO: add torch.jit.script
"""
# erwerfew
# dewf ew
# test 2

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
from args import TestArgs, get_logger
import cv2


@torch.no_grad()
def tester(test_args: argparse, logger):
    logger.info(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
    load_model_path = os.path.join(test_args.exp_path, "model_saved", "model.pth")
    save_output_path = os.path.join(test_args.exp_path, "prediction_saved")
    # 1. -------------Prepare testloader, evaluator---------------------------
    logging.info("testloader, evaluator prepare...")
    test_dataloader = PNGTestloader(image_path=os.path.join(test_args.test_data_path, "image"),
                                    chip_size=test_args.chip_size, stride=test_args.stride,
                                    n_class=test_args.n_class, batch_size=test_args.batch_size,
                                    device=test_args.device)

    evaluator = SegmentationEvaluator(true_label=torch.arange(test_args.n_class))
    logger.info("done")
    # 2. ----------------------------Testing Process and evaluation-------------------------------------
    logger.info("loading model state dict...")
    test_args.model.load_state_dict(torch.load(load_model_path))
    test_args.model.eval()
    test_args.model.to(test_args.device)
    logger.info("done")
    max_batch_num = np.ceil(len(test_dataloader) / test_args.batch_size)
    test_count = 0
    last_batch_flag = False
    with tqdm(total=max_batch_num, unit_scale=True, unit=" batch", colour="magenta", ncols=60) as pbar:
        for i, (data, info) in enumerate(test_dataloader):
            data = data.to(test_args.device)  # data (batch_size, channels, height, width)
            preds = test_args.model(data)  # preds (batch_size, n_class, height, width)
            if i == (max_batch_num - 1):
                last_batch_flag = True
            for whole_label, image_name in test_dataloader.stitcher(preds, info, last_batch_flag):
                # before: whole label (n_class, height, width)
                whole_label = torch.argmax(whole_label, dim=0)
                test_count += 1
                # after: whole label (height, width)
                gt = torch.tensor(cv2.imread(os.path.join(test_args.test_data_path, "gt", image_name))[:, :, 0])
                evaluator.accumulate(whole_label, gt.to(test_args.device))
                cv2.imwrite(os.path.join(save_output_path, image_name), whole_label.cpu().numpy())
            pbar.update()
    evaluator.compute_mean()
    metrics = evaluator.get_metrics()
    logger.info("test miou: {}  test iou: {}".format(metrics["miou"], metrics["iou"]))
    logger.info("count = {}".format(test_count))


if __name__ == "__main__":
    Args = TestArgs()
    logger = get_logger(os.path.join(Args.exp_path, "log_test.txt"))
    logger.info("--------------Test Process----------------\nArgs:")
    for key in Args.origin.keys():
        logger.info(f"  {key}: {Args.origin[key]}")
    logger.info("")
    tester(Args, logger)

