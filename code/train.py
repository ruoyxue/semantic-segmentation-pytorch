"""
It is recommended that data augmentation is finished before training, which means
there wouldn't be any dynamic preprocessing in training process except normalisation.

We provide simple data augmentation code in utils/data_augmentation.py

TODO: add torch.jit.script
TODO: add tensorboard visualisation
TODO: add warm-up to optimizer
TODO: remove ifs to accelerate GPU computations
TODO: add apex.amp
"""

import datetime
import logging
import os.path
import random
import sys
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch import optim, nn
from args import TrainArgs, get_logger
import argparse
from utils import PNGTrainloader, PNGTestloader, \
    SegmentationEvaluator, LogSoftmaxCrossEntropyLoss
from ruamel import yaml
import shutil


def save_script(save_path):
    """ save all .py and .sh documents to save_path/code for reusing """
    shutil.copytree(os.getcwd(), os.path.join(save_path, "code"))
    # for root, dirs, files in os.walk(os.path.join(save_path, "code")):
    #     for file in files:
    #         if file.split(".")[-1] not in ["py", "sh", "yml"]:
    #             os.remove(os.path.join(root, file))
    #     for d in dirs:
    #         if d in ["save", "__pycache__"]:
    #             shutil.rmtree(os.path.join(root, d))


def valider(train_args: argparse, logger):
    """ use train model to test valid data and give metrics """
    logger.info("validation")
    train_args.model.eval()
    test_dataloader = PNGTestloader(image_path=os.path.join(train_args.valid_data_path, "image"),
                                    chip_size=512, stride=256,
                                    n_class=train_args.n_class, batch_size=train_args.batch_size,
                                    device=train_args.device)
    evaluator = SegmentationEvaluator(true_label=torch.arange(train_args.n_class))

    max_batch_num = np.ceil(len(test_dataloader) / train_args.batch_size)
    last_batch_flag = False
    with tqdm(total=max_batch_num, unit_scale=True, unit=" batch", colour="cyan", ncols=80) as pbar:
        for i, (data, info) in enumerate(test_dataloader):
            data = data.to(train_args.device)  # data (batch_size, channels, height, width)
            preds = train_args.model(data)  # preds (batch_size, n_class, height, width)
            if i == (max_batch_num - 1):
                last_batch_flag = True
            for whole_label, image_name in test_dataloader.stitcher(preds, info, last_batch_flag):
                # before: whole label (n_class, height, width)
                whole_label = torch.argmax(whole_label, dim=0)
                # after: whole label (height, width)
                gt = torch.tensor(cv2.imread(os.path.join(train_args.valid_data_path, "gt", image_name))[:, :, 0])
                evaluator.accumulate(whole_label, gt.to(train_args.device))
            pbar.update()
    evaluator.log_metrics()
    return evaluator.metrics["miou"]


def trainer(train_args: argparse, logger):
    if train_args.check_point_mode == "load":
        logger.info("load checkpoint, restarting!")
    save_model_path = os.path.join(train_args.exp_path, "model_saved")
    check_point_path = os.path.join(train_args.exp_path, "checkpoint_saved", "checkpoint.pt")
    # 1. -------------Prepare dataloader, optimizer, scheduler, loss, evaluator---------------------------
    logger.info("prepare dataloader, optimizer, scheduler, loss, evaluator...")
    train_args.model.to(train_args.device)
    train_dataloader = PNGTrainloader(image_path=os.path.join(train_args.train_data_path, "image"),
                                      gt_path=os.path.join(train_args.train_data_path, "gt"),
                                      batch_size=train_args.batch_size, drop_last=True, shuffle=True)
    criterion = LogSoftmaxCrossEntropyLoss(n_class=train_args.n_class, weight=torch.tensor([0.0431, 0.9569]),
                                           smoothing=0.1)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.0862, 1.9138], dtype=torch.float32))
    criterion.to(train_args.device)
    evaluator = SegmentationEvaluator(true_label=torch.arange(train_args.n_class))
    optimizer = optim.SGD(train_args.model.parameters(), lr=train_args.lr, momentum=0.9, weight_decay=5e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5,
                                                     min_lr=1e-6, threshold=1e-3, verbose=True)
    with open(os.path.join(Args.args.exp_path, "config.yml"), "a") as f:
        yaml.dump({"optimizer": {"name": "SGD", "state_dict": optimizer.state_dict()}}, f)
        f.write("\n")
        yaml.dump({"scheduler": {"name": "ReduceLROnPlateau", "state_dict": scheduler.state_dict()}}, f)
        f.write("\n")
    logger.info("done")

    # 2. ---------------------------whether to load checkpoint-------------------------------------------
    start_epoch = 1  # range(start_epoch, epochs + 1) which works for loading checkpoint
    best_valid_miou = 0  # record best valid miou, just save model has the best valid miou
    if train_args.check_point_mode == "load":
        logger.info("load state_dict of model, optimizer, scheduler")
        checkpoint = torch.load(check_point_path)
        train_args.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_valid_miou = checkpoint["best_valid_miou"]
        start_epoch += checkpoint["epoch"]
        logger.info("done")

    # 3. --------------------------Training Process and evaluation----------------------------------------
    train_args.model.train()
    batch_sum = int(len(train_dataloader) / train_args.batch_size)
    for epoch in range(start_epoch, train_args.epochs + 1):
        if (epoch - 1) % 5 == 0:
            logging.info(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
        loss_ = torch.tensor([0], dtype=torch.float32, device=train_args.device,
                             requires_grad=False)
        with tqdm(total=batch_sum, unit_scale=True, unit=" batch",
                  colour="cyan", ncols=80) as pbar:
            for i, (images, gts) in enumerate(train_dataloader):
                # images (batch_size, channel, height, width)
                # gts (batch_size, height, width)
                images = images.to(train_args.device)
                gts = gts.to(train_args.device)
                optimizer.zero_grad()
                # predictions (batch_size, n_class, height, width)
                predictions = train_args.model(images)
                loss = criterion(predictions, gts)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    loss_ += loss
                    evaluator.accumulate(torch.argmax(predictions, dim=1), gts)
                pbar.update()

        loss_ /= batch_sum
        logger.info(f"epoch: {epoch}    loss: {round(loss_.item(), 5)}")
        evaluator.log_metrics()
        evaluator.clear()
        scheduler.step(loss_)
        if epoch % 5 == 0:
            # validation, save model has best valid miou
            with torch.no_grad():
                current_miou = valider(train_args, logger)
                if current_miou > best_valid_miou:
                    best_valid_miou = current_miou
                    # save model
                    torch.save(train_args.model.state_dict(),
                               os.path.join(save_model_path, "model.pth"))
                    logger.info(f"epoch {epoch} best model saved successfully")
                    train_args.model.train()
                    # whether to save checkpoint or not
                    if train_args.check_point_mode in ["save", "load"]:
                        torch.save({
                            "epoch": epoch,
                            "best_valid_miou": best_valid_miou,
                            "model_state_dict": train_args.model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict()
                        }, check_point_path)
                        logger.info(f"epoch {epoch} checkpoint saved successfully")
        logger.info("")


if __name__ == "__main__":
    Args = TrainArgs()
    torch.manual_seed(Args.args.random_seed)
    logger = get_logger(os.path.join(Args.args.exp_path, "log_train.txt"))
    logger.info("record experiment code, config...")
    if Args.args.check_point_mode != "load":
        save_script(Args.args.exp_path)
        with open(os.path.join(Args.args.exp_path, "config.yml"), "a") as f:
            yaml.dump({"args": Args.origin}, f, Dumper=yaml.RoundTripDumper)
            f.write("\n")
    logger.info("done")
    logger.info(Args.origin)
    trainer(Args.args, logger)



