import datetime
import logging
from tqdm import tqdm
import torch
from torch import optim, nn
import torch.nn.functional as F
from Args import Args, Train_Args
from torch.utils.data import DataLoader
import argparse
import os
from utils.Evaluator import Classification_Evaluator
from torchvision import datasets, transforms

def Trainer(args : argparse):
    logging.info(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
    # 1. -------------Prepare dataset, dataloader, optimizer, scheduler, loss, evaluator---------------------------
    # dataset = TrainDataset()
    # dataiter = DataLoader(dataset)
    logging.info(args.model)
    traindata_dataloader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../dataset/fashionmnist_data/', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ])),
        batch_size=args.batchsize, shuffle=True, drop_last=True)
    criterion = nn.CrossEntropyLoss()
    evaluator = Classification_Evaluator(true_label=range(10))
    optimizer = optim.SGD(args.model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.25, threshold=5e-3,
                                                     patience=1, min_lr=1e-5, verbose=True)
    args.model.to(args.device)

    # 2. ----------------whether load checkpoint or not--------------------------------------------------
    start_epoch = 1  # range(start_epoch, epochs+1) which works for loading checkpoint
    if args.check_point_mode == "load":
        checkpoint = torch.load(args.check_point_path)
        args.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch += checkpoint["epoch"]

    # 3. ----------------------------Training Process and evaluation-------------------------------------
    args.model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        loss_ = torch.tensor([0], dtype=torch.float32, device=args.device)
        with tqdm(total=60000 // args.batchsize, unit_scale=True, unit=" img", colour="cyan", ncols=60) as pbar:
            for batch_num, (images, labels) in enumerate(traindata_dataloader):
                images = images.to(args.device)
                labels = labels.to(args.device)
                optimizer.zero_grad()
                preds = args.model(images)
                loss = criterion(preds, labels)
                loss_ += loss
                evaluator.accumulate(torch.argmax(preds, dim=1), labels)
                loss.backward()
                optimizer.step()
                pbar.update(1)

        logging.info(f"epoch: {epoch}    loss: {round(loss_.item() / (batch_num + 1), 5)}")
        evaluator.log_metrics()
        evaluator.clear()
        scheduler.step(loss_)
        if epoch % 5 == 0:
            # torch.save(args.model.state_dict(),
            #            os.path.join(args.save_model_path, "model_epoch_{}.pth".format(epoch)))
            logging.info(f"\nepoch {epoch} model saved successfully\n")

            # whether to save checkpoint or not
            if args.check_point_mode in ["save", "load"]:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": args.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict" : scheduler.state_dict()
                }, args.check_point_path)
                logging.info(f"epoch {epoch} checkpoint saved successfully\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = Train_Args()
    Trainer(args.args)

