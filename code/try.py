import logging
import os
import random
import shutil
from torchvision import transforms, utils as vutils
import numpy as np
import torch
import utils.preprocessing as prepro
import torch.nn.functional as F
from torch import nn
from collections import namedtuple
import time
import argparse
import datetime
import cv2
from tqdm import tqdm
from utils import PNGTestloader
from utils import PNGTrainloader, TIFFTrainloader, PlateauLRScheduler, LogSoftmaxCELoss
from ruamel import yaml
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from models import UNet


image_path = "/home/xueruoyao/Documents/PythonProgram/dataset/deepglobe/image"
gt_path = "/home/xueruoyao/Documents/PythonProgram/dataset/deepglobe/gt"

# from utils import ComputerVisionTestLoader
#
# test_loader = ComputerVisionTestLoader(image_path, 600, 256)
# print(test_loader.chip_info)

# device = torch.device("cuda:0")
# test_loader = PNGTestloader(image_path, chip_size=256, stride=128, n_class=3,
#                             batch_size=4, device=device)
# max_batch = np.ceil(len(test_loader) / 1)
# print(max_batch)
# last_batch_flag = False
# for i, (data, info) in enumerate(test_loader):
#     print(i)
#     if i == max_batch-1:
#         last_batch_flag = True
#     data = data.to(device)
#     for whole_image, image_name in test_loader.stitcher(data, info, last_batch_flag):
#         whole_image = whole_image.permute([1, 2, 0])
#         whole_image = (whole_image.detach().cpu().numpy())
#         cv2.imshow("whole image", np.uint8(whole_image))
#         cv2.waitKey(0)



# from torchvision import transforms
#
# transform = nn.Sequential(transforms.RandomCrop((512, 512)),
#                           transforms.ToTensor)
#
# for image_name in os.listdir(image_path):
#     image = cv2.imread(os.path.join(image_path, image_name))
#     cv2.imshow("before", image)
#     im_data = image.transpose((2, 0, 1))
#     im_data = torch.from_numpy(im_data)
#     im_data = transform(im_data)
#     print((im_data).dtype)
#     image = np.rollaxis(np.array(im_data), 0, 3)
#     cv2.imshow("after", image)
#     print(image.shape)
#     cv2.waitKey(0)

# from smalltool import compute_rgb_mean_std
# compute_rgb_mean_std(image_path)


# from utils.data_augmentation import random_mosaic
#
# image_list, label_list = [], []
# for image_name in os.listdir(image_path)[4:8]:
#     image_list.append(cv2.imread(os.path.join(image_path, image_name)))
#     label_list.append(cv2.imread(os.path.join(label_path, image_name)))
# image, label = random_mosaic(image_list, label_list, 512)
# cv2.imshow("image", image)
# cv2.imshow("label", label * 10)
# tcv2.waitKey(0)


# trans = transforms.Compose([transforms.ToTensor(),
#                             transforms.Normalize(mean=(89.8, 91.3, 89.9), std=(68.0, 65.6, 66.4))])
#
# for image_name in os.listdir(image_path):
#     # image = Image.open(os.path.join(image_path, image_name)).convert("RGB")
#     image = cv2.imread(os.path.join(image_path, image_name))
#     # image = torch.from_numpy(image).float()
#     image = trans(image)
#     print(image.shape)
#     print((image > 0).any())
#
#     break
#
# image_path = "/home/xueruoyao/Documents/PythonProgram/MyFramework/code/save/prediction_saved/FCN8s"
# print(len(os.listdir(image_path)))

# info_dict = {
#     "school": "2",
#     "list": ["1", 2, 3, 4]
# }
#
# with open("./test.yml", "w") as f:
#     yaml.dump(info_dict, f)
# with open("test.yml", "r") as f:
#     dict = yaml.load(f.read(), Loader=yaml.Loader)
# print(type(dict["list"]))
# shutil.copytree(os.getcwd(), "../experiment")
# for root, dirs, files in os.walk("../experiment"):
#     for file in files:
#         if file.split(".")[-1] not in ["py", "sh"]:
#             os.remove(os.path.join(root, file))
#
#     for dir in dirs:
#         if dir not in ["utils", "block", "models"]:
#             shutil.rmtree(os.path.join(root, dir))
#

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# terminal_handler = logging.StreamHandler()
# terminal_handler.setFormatter(logging.Formatter("%(message)s"))
# terminal_handler.setLevel(logging.DEBUG)
# file_handler = logging.FileHandler(filename="log.txt")
# file_handler.setFormatter(logging.Formatter("[%(asctime)s]\n%(message)s"))
# file_handler.setLevel(logging.INFO)
#
# logger.addHandler(terminal_handler)
# logger.addHandler(file_handler)
# logging.info("info hello")

# model = nn.Sequential(nn.Linear(3, 64),
#                       nn.Linear(64, 128))
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
# optimizer2 = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-5)
# scheduler = PlateauLRScheduler(optimizer, patience=2, min_lr=1e-6, lr_factor=0.5,
#                                warmup_duration=20)
# scheduler2 = PlateauLRScheduler(optimizer2, patience=5, min_lr=1e-6, lr_factor=0.25,
#                                warmup_duration=20)

# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2)
#
# optimizer.param_groups[0]["lr"] = 104
# print(optimizer.state_dict())

# print(scheduler.get_lr())

# logging.basicConfig(level=logging.DEBUG, format="%(message)s")
#
# scheduler2.load_state_dict(scheduler.state_dict())

# loss_ = torch.tensor([20], dtype=torch.float32).cuda()
# for epoch in range(1, 1000):
#     if epoch % 8 == 0:
#         loss_ /= 2
#     scheduler.step(loss_, epoch)
# print(scheduler.best_metric)

# scheduler2.load_state_dict(scheduler.state_dict())

# criterion = LogSoftmaxCELoss(n_class=2, weight=torch.tensor([0.0431, 0.9569]),
#                              smoothing=0.002)
# criterion2 = LogSoftmaxCELoss(n_class=2, weight=torch.tensor([2, 3]),
#                               smoothing=0.232)
# print(optimizer.state_dict())
# print(scheduler.state_dict())
# # print(criterion2.state_dict())
# criterion2.load_state_dict(criterion.state_dict())
# print(criterion2.state_dict())

# writer = SummaryWriter("./tensorboard_info/")

# x = torch.tensor([5], dtype=torch.float32)
# y = torch.tensor([2], dtype=torch.float32)
# model = nn.Linear(1, 1)
# criterion = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
# for epoch in range(500):
#     output = model(x)
#     loss = criterion(output, x)
#     writer.add_scalar("loss", loss, epoch)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# writer.flush()

#
#
# loss = 1
# acc = 0
# for epoch in range(1000):
#     time.sleep(1)
#     loss += 8
#     acc += 10
#     print(epoch)
#     writer.add_scalars("metrics", {
#         "loss": loss,
#         "acc": acc
#     }, epoch)
#
#     writer.flush()
#
#
# writer.close()
# a = torch.tensor([1, 2], dtype=torch.float32).cuda()
# l = [a[i].item() for i in range(2)]
# print(l)
# print(list(a.cpu().numpy()))
# print(type(optimizer.state_dict()["param_groups"]))
# a = [a[i].item() for i in range(2)]
# print(type(a))
# with open(os.path.join("config.yml"), "w") as f:
#     yaml.dump({"args": a}, f)
#     f.write("\n")
#
# a = torch.from_numpy(np.array(a)).float().cuda()
# print(a)
#
# augmentation = prepro.ProcessingSequential([
#     prepro.RandomCrop(chip_size=512),
#     prepro.RandomRotate(random_choice=[0, 90, 180, 270]),
#     prepro.RandomFlip(random_choice=[-1, 0, 1]),
#     prepro.Normalize(mean=(73.4711, 97.6228, 104.4753), std=(31.2603, 32.3015, 39.8499)),
#     prepro.ToTensor()
# ])
#
# print(augmentation.list_of_repr())

# output_string = augmentation.__class__.__name__ + "("
# for key in augmentation.__dict__.keys():
#     output_string += key + "=" + str(augmentation.__dict__[key]) + ", "
#
# output_string = output_string.strip(", ") + ")"
# print(output_string)
# with tqdm(total=len(os.listdir(image_path))) as pbar:
#     for image_name in os.listdir(image_path):
#         image = cv2.imread(os.path.join(image_path, image_name))
#         gt = cv2.imread(os.path.join(gt_path, image_name))
#         # cv2.imshow("image_before", image)
#         # cv2.imshow("gt_before", np.uint8(gt * 255))
#         image_after, gt_after = augmentation(image, gt)
#         # cv2.imshow("image_after", image_after)
#         # cv2.imshow("gt_after", np.uint8(gt_after * 255))
#         # cv2.waitKey(0)
#         pbar.update()


#
# random_mosaic = prepro.RandomMosaic(final_size=512, n_channel=3)
# image_list = []
# gt_list = []
# for image_name in os.listdir(image_path):
#     image = cv2.imread(os.path.join(image_path, image_name))
#     gt = cv2.imread(os.path.join(gt_path, image_name))
#     image_list.append(image)
#     gt_list.append(gt[:, :, 0])
#     print("OK")
#     if len(image_list) == 4:
#         image, gt = random_mosaic(image_list, gt_list)
#         cv2.imshow("image", image)
#         cv2.imshow("gt", np.uint8(gt * 255))
#         cv2.waitKey(0)
#         image_list.clear()
#         gt_list.clear()
