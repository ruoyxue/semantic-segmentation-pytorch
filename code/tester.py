"""
class for testing
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
from args import TestArgs, set_logger
import cv2


class Tester:
    def __init__(self, test_args: argparse):
        self.test_args = test_args
        self.load_model_path = os.path.join(self.test_args.exp_path, "model_saved", "model.pth")
        self.save_preds_path = os.path.join(self.test_args.exp_path, "prediction_saved")
        self.testloader, self.evaluator = None, None
        self.init_elements()
        self.load_model_state()
        self.test()

    def init_elements(self):
        """ init testloader, evaluator """
        logging.info("init testloader, evaluator...")
        self.testloader = PNGTestloader(image_path=os.path.join(self.test_args.test_data_path, "image"),
                                        chip_size=self.test_args.chip_size, stride=self.test_args.stride,
                                        n_class=self.test_args.n_class, batch_size=self.test_args.batch_size,
                                        device=self.test_args.device)
        self.evaluator = SegmentationEvaluator(true_label=torch.arange(self.test_args.n_class))

    def load_model_state(self):
        """ load model state dict """
        logging.info("load model state dict...")
        self.test_args.model.load_state_dict(torch.load(self.load_model_path))
        self.test_args.model.eval()
        self.test_args.model.to(self.test_args.device)

    def test(self):
        """ function for testing """
        max_batch_num = np.ceil(len(self.testloader) / self.test_args.batch_size)
        test_count = 0
        last_batch_flag = False
        with tqdm(total=max_batch_num, unit_scale=True, unit=" batch", colour="magenta", ncols=60) as pbar:
            for i, (data, info) in enumerate(self.testloader):
                data = data.to(self.test_args.device)  # data (batch_size, channels, height, width)
                preds = self.test_args.model(data)  # preds (batch_size, n_class, height, width)
                if i == (max_batch_num - 1):
                    last_batch_flag = True
                for whole_label, image_name in self.testloader.stitcher(preds, info, last_batch_flag):
                    # before: whole label (n_class, height, width)
                    whole_label = torch.argmax(whole_label, dim=0)
                    test_count += 1
                    # after: whole label (height, width)
                    gt = torch.tensor(cv2.imread(os.path.join(self.test_args.test_data_path, "gt", image_name))[:, :, 0])
                    self.evaluator.accumulate(whole_label, gt.to(self.test_args.device))
                    cv2.imwrite(os.path.join(self.save_preds_path, image_name), whole_label.cpu().numpy())
                pbar.update()
        self.evaluator.compute_mean()
        metrics = self.evaluator.get_metrics()
        logging.info("test miou: {}  test iou: {}".format(metrics["miou"], metrics["iou"]))
        logging.info("count = {}".format(test_count))


if __name__ == "__main__":
    Args = TestArgs()
    set_logger(os.path.join(Args.exp_path, "log_test.txt"))
    logging.info("--------------Test Process----------------\nArgs:")
    for key in Args.origin.keys():
        logging.info(f"  {key}: {Args.origin[key]}")
    logging.info("\n{}".format(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")))
    Tester(Args)
