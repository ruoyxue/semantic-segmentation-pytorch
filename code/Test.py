import logging
import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
from utils.Evaluator import Classification_Evaluator
from torchvision import datasets, transforms
from Args import Args, Test_Args

def Tester(args : argparse):
    logging.info(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
    # 1. -------------Prepare dataset, dataloader, evaluator---------------------------
    # dataset = TrainDataset()
    # dataiter = DataLoader(dataset)
    logging.debug(args.model)
    testdata_iter = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../dataset/fashionmnist_data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.batchsize)
    evaluator = Classification_Evaluator(labels=range(10))

    # 2. ----------------------------Testing Process and evaluation-------------------------------------
    args.model.load_state_dict(torch.load(args.load_model_path))
    args.model.eval()
    args.model.to(args.device)
    with tqdm(total=10000 // args.batchsize, unit_scale=True, unit=" img", colour="cyan", ncols=60) as pbar:
        for batch_num, (images, labels) in enumerate(testdata_iter):
            images = images.to(args.device)
            labels = labels.to(args.device)
            preds = args.model(images)

            evaluator.accumulate(torch.argmax(preds, dim=1), labels)
            pbar.update(1)

    logging.info(evaluator.count)
    evaluator.log_metrics()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = Test_Args()
    Tester(args.args)

