"""
It is recommended that data augmentation is finished before training, which means
there wouldn't be any dynamic preprocessing in training process except normalisation.

We provide simple data augmentation code in utils/data_augmentation.py

TODO: add torch.jit.script
TODO: add tensorboard visualisation
TODO: remove ifs to accelerate GPU computations
TODO: add apex.amp
TODO: replace python code with C++, python serves as glue
TODO: add data parallel
TODO: add distributive training
TODO: add gradient accumulation to upgrade gradients after a few epochs
      (use small batchsize to simulate larger one)
TODO: add pretrained parameters to models
TODO: replace static data augmentation with dynamic ones, and pack functions in .py file into a class
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
    SegmentationEvaluator, LogSoftmaxCELoss, PlateauLRScheduler
from ruamel import yaml
import shutil
from torch.utils.tensorboard import SummaryWriter


def valider(train_args: argparse, logger):
    """ use train model to test valid data and give metrics """
    train_args.model.eval()
    test_dataloader = PNGTestloader(image_path=os.path.join(train_args.valid_data_path, "image"),
                                    chip_size=train_args.chip_size, stride=train_args.stride,
                                    n_class=train_args.n_class, batch_size=train_args.batch_size,
                                    device=train_args.device)
    evaluator = SegmentationEvaluator(true_label=torch.arange(train_args.n_class))

    max_batch_num = np.ceil(len(test_dataloader) / train_args.batch_size)
    last_batch_flag = False
    with tqdm(total=max_batch_num, unit_scale=True, unit=" batch", colour="magenta", ncols=60) as pbar:
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
    evaluator.compute_mean()
    train_args.model.train()
    return evaluator.get_metrics()


def trainer(train_args: argparse, logger):
    save_model_path = os.path.join(train_args.exp_path, "model_saved")
    save_checkpoint_path = os.path.join(train_args.exp_path, "checkpoint_saved", "checkpoint.pt")
    save_tensorboard_path = os.path.join(train_args.exp_path, "tensorboard_saved")
    # 1. -------------Prepare dataloader, optimizer, scheduler, loss, evaluator---------------------------
    logger.info("prepare dataloader, optimizer, scheduler, loss, evaluator...")
    train_args.model.to(train_args.device)
    train_dataloader = PNGTrainloader(image_path=os.path.join(train_args.train_data_path, "image"),
                                      gt_path=os.path.join(train_args.train_data_path, "gt"),
                                      batch_size=train_args.batch_size, drop_last=True, shuffle=True,
                                      chip_size=train_args.chip_size)
    criterion = LogSoftmaxCELoss(n_class=train_args.n_class, weight=torch.tensor([0.0431, 0.9569]),
                                 smoothing=0.002)
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.0862, 1.9138], dtype=torch.float32))
    criterion.to(train_args.device)
    evaluator = SegmentationEvaluator(true_label=torch.arange(train_args.n_class))
    optimizer = optim.SGD(train_args.model.parameters(), lr=train_args.lr, momentum=0.9, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5,
    #                                                  min_lr=1e-6, threshold=1e-3, verbose=True)
    scheduler = PlateauLRScheduler(optimizer, mode="min", lr_factor=0.5, patience=2, min_lr=1e-6,
                                   threshold=1e-2, warmup_duration=20)
    if train_args.check_point_mode == "save":
        with open(os.path.join(train_args.exp_path, "config.yml"), "a") as f:
            yaml.dump({"optimizer": {"type": str(type(optimizer)), "state_dict": optimizer.state_dict()}}, f)
            f.write("\n")
            yaml.dump({"scheduler": scheduler.state_dict()}, f, Dumper=yaml.RoundTripDumper)
            f.write("\n")
            yaml.dump({"loss": criterion.state_dict()}, f)
            f.write("\n")
            yaml.dump({"trainloader": train_dataloader.state_dict()}, f, Dumper=yaml.RoundTripDumper)
            f.write("\n")
    logger.info("done")

    # 2. ---------------------------whether to load checkpoint-------------------------------------------
    start_epoch = 1  # range(start_epoch, epochs + 1) which works for loading checkpoint
    best_valid_metric = 0  # record best valid metric
    best_valid_epoch = 1
    if train_args.check_point_mode == "load":
        logger.info("load state_dict of model, optimizer, scheduler, loss")
        checkpoint = torch.load(save_checkpoint_path)
        train_args.model.load_state_dict(checkpoint["model_state_dict"])
        criterion.load_state_dict(checkpoint["criterion_state_dict"])
        criterion.to(train_args.device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # checkpoint["scheduler_state_dict"]["patience"] = 2
        # checkpoint["scheduler_state_dict"]["min_lr"] = 1e-6
        # checkpoint["scheduler_state_dict"]["threshold"] = 1e-2
        # checkpoint["scheduler_state_dict"]["warmup_duration"] = 20
        # logging.info("-----------revise scheduler patience=2, min_lr=1e-6, "
        #              "threshold=1e-2, warmup_duration=20-----------")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_valid_metric = checkpoint["best_valid_metric"]
        best_valid_epoch = checkpoint["best_valid_epoch"]
        start_epoch += checkpoint["epoch"]
        logger.info("done")

    # 3. --------------------------Training Process and evaluation----------------------------------------
    train_args.model.train()
    writer_metrics = SummaryWriter(os.path.join(save_tensorboard_path, "metrics"))
    writer_model = SummaryWriter(os.path.join(save_tensorboard_path, "model"))
    batch_sum = int(len(train_dataloader) / train_args.batch_size)
    for epoch in range(start_epoch, train_args.epochs + 1):
        if (epoch - 1) % 5 == 0:
            logging.info(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
        logger.info(f"epoch {epoch}")
        loss_ = torch.tensor([0], dtype=torch.float32, device=train_args.device, requires_grad=False)
        with tqdm(total=batch_sum, unit_scale=True, unit=" batch", colour="cyan", ncols=80) as pbar:
            for i, (images, gts) in enumerate(train_dataloader):
                # images (batch_size, channel, height, width)
                # gts (batch_size, height, width)
                images = images.to(train_args.device)
                if i == 0 and epoch == 1:
                    train_args.model.eval()
                    writer_model.add_graph(train_args.model, images)
                    writer_model.close()
                    train_args.model.train()
                gts = gts.to(train_args.device)
                optimizer.zero_grad()
                # predictions (batch_size, n_class, height, width)
                predictions = train_args.model(images)
                loss = criterion(predictions, gts)
                loss.backward()
                optimizer.step()
                loss_ += loss
                evaluator.accumulate(torch.argmax(predictions, dim=1), gts)
                pbar.update()

        loss_ /= batch_sum
        evaluator.compute_mean()
        train_metrics = evaluator.get_metrics()
        logger.info("train_loss:{}  train_iou: {}  current_lr: {}".format(
            round(loss_.item(), 5),
            train_metrics["iou"],
            scheduler.get_lr()
        ))
        scheduler.step(loss_, epoch)
        logger.info("")
        if epoch % 1 == 0:
            # validation, save model has best valid miou
            with torch.no_grad():
                valid_metrics = valider(train_args, logger)
                if valid_metrics["iou"] > best_valid_metric:
                    best_valid_metric = valid_metrics["iou"]
                    best_valid_epoch = epoch
                logger.info("valid iou: {}  best iou: {}  best valid epoch: {}" .
                            format(valid_metrics["iou"], best_valid_metric, best_valid_epoch))
                if valid_metrics["iou"] == best_valid_metric:
                    # save model
                    torch.save(train_args.model.state_dict(),
                               os.path.join(save_model_path, "model.pth"))
                    # whether to save checkpoint or not
                    if train_args.check_point_mode in ["save", "load"]:
                        torch.save({
                            "epoch": epoch,
                            "best_valid_metric": best_valid_metric,
                            "best_valid_epoch": best_valid_epoch,
                            "model_state_dict": train_args.model.state_dict(),
                            "criterion_state_dict": criterion.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict()
                        }, save_checkpoint_path)
                        logger.info(f"--------------- epoch {epoch} model, checkpoint "
                                    f"saved successfully ---------------")
        logger.info("")
        writer_metrics.add_scalars("", {
            "train_loss": round(loss_.item(), 5),
            "train_iou": train_metrics["iou"],
            "valid_iou": valid_metrics["iou"]
        }, epoch)
        writer_metrics.flush()
        evaluator.clear()
    writer_metrics.close()


if __name__ == "__main__":
    Args = TrainArgs()
    torch.manual_seed(Args.random_seed)
    logger = get_logger(os.path.join(Args.exp_path, "log_train.txt"))
    # record experiment information
    if Args.check_point_mode == "save":
        logger.info("--------------Train Process----------------\nArgs:")
        shutil.copytree(os.getcwd(), os.path.join(Args.exp_path, "code"))
        with open(os.path.join(Args.exp_path, "config.yml"), "a") as f:
            yaml.dump({"args": Args.origin}, f, Dumper=yaml.RoundTripDumper)
            f.write("\n")
    elif Args.check_point_mode == "load":
        logger.info("-------------Retrain Process--------------\n"
                    "------------Load Checkpoint, Restarting----------\nArgs:")
    for key in Args.origin.keys():
        logger.info(f"  {key}: {Args.origin[key]}")
    logger.info("")
    trainer(Args, logger)

