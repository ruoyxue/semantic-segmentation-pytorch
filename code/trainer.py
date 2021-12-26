"""
class for training

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
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch import optim, nn
import argparse
from utils import PNGTrainloader, PNGTestloader, \
    SegmentationEvaluator, LogSoftmaxCELoss, PlateauLRScheduler
from ruamel import yaml
import shutil
from torch.utils.tensorboard import SummaryWriter
from args import TrainArgs, set_logger


class Trainer:
    def __init__(self, train_args: argparse):
        self.train_args = train_args
        self.save_model_path = os.path.join(train_args.exp_path, "model_saved")
        self.save_checkpoint_path = os.path.join(train_args.exp_path, "checkpoint_saved", "checkpoint.pt")
        self.save_tensorboard_path = os.path.join(train_args.exp_path, "tensorboard_saved")
        self.start_epoch = 1  # [start_epoch, epochs] which works for loading checkpoint
        self.best_valid_metric, self.best_valid_epoch = 0, 1  # record best valid info
        self.trainloader, self.criterion, self.evaluator, self.optimizer, self.scheduler \
            = None, None, None, None, None
        self.save_interval = 5  # intervals between two validations
        self.init_elements()
        self.init_checkpoint_mode()
        self.train()

    def init_elements(self):
        """ init dataloader, criterion, evaluator, optimizer, scheduler """
        logging.info("init trainloader, criterion, evaluator, optimizer, scheduler")
        self.train_args.model.to(self.train_args.device)
        self.trainloader = PNGTrainloader(image_path=os.path.join(self.train_args.train_data_path, "image"),
                                          gt_path=os.path.join(self.train_args.train_data_path, "gt"),
                                          batch_size=self.train_args.batch_size, drop_last=True, shuffle=True,
                                          chip_size=self.train_args.chip_size)
        # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.0862, 1.9138], dtype=torch.float32))
        self.criterion = LogSoftmaxCELoss(n_class=self.train_args.n_class, weight=torch.tensor([0.0431, 0.9569]))
        self.criterion.to(self.train_args.device)
        self.evaluator = SegmentationEvaluator(true_label=torch.arange(self.train_args.n_class))
        # optimizer = optim.SGD(train_args.model.parameters(), lr=train_args.lr, momentum=0.9, weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(self.train_args.model.parameters(), lr=self.train_args.lr,
                                          betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5,
        #                                                  min_lr=1e-6, threshold=1e-3, verbose=True)
        self.scheduler = PlateauLRScheduler(self.optimizer, mode="min", lr_factor=0.5, patience=20,
                                            min_lr=1e-6, threshold=5e-4, warmup_duration=30)

    def load_checkpoints(self):
        """ load checkpoints """
        logging.info("load state_dict of model, optimizer, scheduler, criterion")
        checkpoint = torch.load(self.save_checkpoint_path)
        self.train_args.model.load_state_dict(checkpoint["model_state_dict"])
        self.criterion.load_state_dict(checkpoint["criterion_state_dict"])
        self.criterion.to(self.train_args.device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        checkpoint["scheduler_state_dict"]["patience"] = 20
        checkpoint["scheduler_state_dict"]["current_lr"] = 0.001
        # checkpoint["scheduler_state_dict"]["min_lr"] = 1e-6
        # checkpoint["scheduler_state_dict"]["threshold"] = 1e-3
        # checkpoint["scheduler_state_dict"]["warmup_duration"] = 50
        logging.info("----------- revise scheduler patience=20,current_lr=0.001 -----------")
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_valid_metric = checkpoint["best_valid_metric"]
        self.best_valid_epoch = checkpoint["best_valid_epoch"]
        self.start_epoch += checkpoint["epoch"]

    def init_checkpoint_mode(self):
        """ init in terms of checkpoint mode; Save code and config for 'save', while load checkpoints for 'load' """
        if self.train_args.check_point_mode == "save":  # save code and config.yml
            logging.info("--------------Train Process----------------\nArgs:")
            for key in self.train_args.origin.keys():
                logging.info(f"  {key}: {self.train_args.origin[key]}")
            shutil.copytree(os.getcwd(), os.path.join(self.train_args.exp_path, "code"))
            self.save_config()
        elif self.train_args.check_point_mode == "load":  # load checkpoints
            logging.info("\n-------------Retrain Process--------------\n"
                         "------------Load Checkpoint, Restarting----------\nArgs:")
            self.load_checkpoints()
            for key in self.train_args.origin.keys():
                logging.info(f"  {key}: {self.train_args.origin[key]}")
            logging.info("")

    def save_config(self):
        """ write state dicts info in config.yml """
        with open(os.path.join(self.train_args.exp_path, "config.yml"), "a") as f:
            yaml.dump({"args": self.train_args.origin}, f, Dumper=yaml.RoundTripDumper)
            f.write("\n")
            yaml.dump({"trainloader": self.trainloader.state_dict()}, f, Dumper=yaml.RoundTripDumper)
            f.write("\n")
            yaml.dump({"optimizer": {"type": str(type(self.optimizer)), "state_dict": self.optimizer.state_dict()}}, f)
            f.write("\n")
            yaml.dump({"scheduler": self.scheduler.state_dict()}, f, Dumper=yaml.RoundTripDumper)
            f.write("\n")
            yaml.dump({"loss": self.criterion.state_dict()}, f)
            f.write("\n")

    def train(self):
        """ training for multiple epochs """
        # display model in tensorboard
        writer_model = SummaryWriter(os.path.join(self.save_tensorboard_path, "model"))
        self.train_args.model.eval()
        tem = torch.zeros((self.train_args.batch_size, 3, 256, 256), dtype=torch.float32)
        writer_model.add_graph(
            self.train_args.model,
            tem.to(self.train_args.device)
            )
        writer_model.close()
        self.train_args.model.train()

        writer_metrics = SummaryWriter(os.path.join(self.save_tensorboard_path, "metrics"))
        batch_sum = int(len(self.trainloader) / self.train_args.batch_size)
        for epoch in range(self.start_epoch, self.train_args.epochs + 1):
            if (epoch - 1) % 5 == 0:
                logging.info(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
            loss, train_metrics = self.single_epoch_train(epoch, batch_sum)
            if epoch % self.save_interval == 0:
                with torch.no_grad():
                    valid_metrics = self.validation()
                    if valid_metrics["iou"] > self.best_valid_metric:
                        self.best_valid_metric = valid_metrics["iou"]
                        self.best_valid_epoch = epoch
                    logging.info("valid iou: {}  best iou: {}  best valid epoch: {}" .
                                 format(valid_metrics["iou"], self.best_valid_metric, self.best_valid_epoch))
                    if valid_metrics["iou"] == self.best_valid_metric:
                        self.save_model_checkpoint(epoch)
                    logging.info("")
                    writer_metrics.add_scalars("", {"valid_iou": valid_metrics["iou"]}, epoch)
            writer_metrics.add_scalars("", {
                "train_loss": round(loss, 5),
                "train_iou": train_metrics["iou"],
            }, epoch)
            writer_metrics.flush()
            self.evaluator.clear()
        writer_metrics.close()

    def single_epoch_train(self, epoch: int, batch_sum: int):
        """ train for a single epoch, used in train function"""
        logging.info(f"epoch {epoch}")
        loss_ = torch.tensor([0], dtype=torch.float32, device=self.train_args.device, requires_grad=False)
        with tqdm(total=batch_sum, unit_scale=True, unit=" batch", colour="cyan", ncols=80) as pbar:
            for i, (images, gts) in enumerate(self.trainloader):
                # images (batch_size, channel, height, width), gts (batch_size, height, width)
                images = images.to(self.train_args.device)
                gts = gts.to(self.train_args.device)
                self.optimizer.zero_grad()
                # predictions (batch_size, n_class, height, width)
                predictions = self.train_args.model(images)
                loss = self.criterion(predictions, gts)
                loss.backward()
                self.optimizer.step()
                loss_ += loss
                self.evaluator.accumulate(torch.argmax(predictions, dim=1), gts)
                pbar.update()
        self.evaluator.compute_mean()
        loss_ /= batch_sum
        train_metrics = self.evaluator.get_metrics()
        logging.info("train_loss:{}  train_iou: {}  current_lr: {}".format(
            round(loss_.item(), 5),
            train_metrics["iou"],
            self.scheduler.get_lr()
        ))
        self.scheduler.step(loss_, epoch)
        logging.info("")
        return loss_.item(), train_metrics

    def validation(self):
        """ validation in train function, testing valid data and give metrics """
        self.train_args.model.eval()
        test_dataloader = PNGTestloader(image_path=os.path.join(self.train_args.valid_data_path, "image"),
                                        chip_size=self.train_args.chip_size, stride=self.train_args.stride,
                                        n_class=self.train_args.n_class, batch_size=self.train_args.batch_size,
                                        device=self.train_args.device)
        evaluator = SegmentationEvaluator(true_label=torch.arange(self.train_args.n_class))

        max_batch_num = np.ceil(len(test_dataloader) / self.train_args.batch_size)
        last_batch_flag = False
        with tqdm(total=max_batch_num, unit_scale=True, unit=" batch", colour="magenta", ncols=60) as pbar:
            for i, (data, info) in enumerate(test_dataloader):
                data = data.to(self.train_args.device)  # data (batch_size, channels, height, width)
                preds = self.train_args.model(data)  # preds (batch_size, n_class, height, width)
                if i == (max_batch_num - 1):
                    last_batch_flag = True
                for whole_label, image_name in test_dataloader.stitcher(preds, info, last_batch_flag):
                    # before: whole label (n_class, height, width)
                    whole_label = torch.argmax(whole_label, dim=0)
                    # after: whole label (height, width)
                    gt = torch.tensor(cv2.imread(
                        os.path.join(self.train_args.valid_data_path, "gt", image_name))[:, :, 0])
                    evaluator.accumulate(whole_label, gt.to(self.train_args.device))
                pbar.update()
        evaluator.compute_mean()
        self.train_args.model.train()
        return evaluator.get_metrics()

    def save_model_checkpoint(self, epoch: int):
        """ save model and checkpoint of the best valid metrics """
        torch.save(self.train_args.model.state_dict(),
                   os.path.join(self.save_model_path, "model.pth"))
        if self.train_args.check_point_mode in ["save", "load"]:
            torch.save({
                "epoch": epoch,
                "best_valid_metric": self.best_valid_metric,
                "best_valid_epoch": self.best_valid_epoch,
                "model_state_dict": self.train_args.model.state_dict(),
                "criterion_state_dict": self.criterion.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
            }, self.save_checkpoint_path)
            logging.info(f"--------------- epoch {epoch} model, checkpoint saved successfully ---------------")


if __name__ == "__main__":
    Args = TrainArgs()
    torch.manual_seed(Args.random_seed)
    set_logger(os.path.join(Args.exp_path, "log_train.txt"))
    Trainer(Args)
