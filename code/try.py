import logging
import os
import random
import shutil

from torchvision import transforms, utils as vutils
import numpy as np
import torch
from torch import nn
from collections import namedtuple
import time
import datetime
import cv2
from utils import PNGTestloader
from utils import PNGTrainloader, TIFFTrainloader
from ruamel import yaml
image_path = "/home/xueruoyao/Documents/PythonProgram/MyFramework/dataset/semantic_segmentation/original/image"
label_path = "/home/xueruoyao/Documents/PythonProgram/MyFramework/dataset/semantic_segmentation/original/label"

# from utils import ComputerVisionTestLoader
#
# test_loader = ComputerVisionTestLoader(image_path, 600, 256)
# print(test_loader.chip_info)

# device = torch.device("cuda:0")
# test_loader = PNGTestloader(image_path, chip_size=256, stride=128, n_class=3,
#                             batch_size=11, device=device)
# max_batch = np.ceil(len(test_loader) / 11)
# print(max_batch)
# last_batch_flag = False
# for i, (data, info) in enumerate(test_loader):
#     print(i)
#     if i == max_batch-1:
#         last_batch_flag = True
#     data = data.to(device)
#     for whole_image, image_name in test_loader.stitcher(data, info, last_batch_flag):
#         if i == max_batch -1:
#             whole_image = whole_image.permute([1, 2, 0])
#             whole_image = (whole_image.detach().cpu().numpy())
#



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

info_dict = {
    "school": "2",
    "list": ["1", 2, 3, 4]
}
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
print(torch.cuda.is_available())
a = torch.tensor(5)
a.cuda()
print(a)