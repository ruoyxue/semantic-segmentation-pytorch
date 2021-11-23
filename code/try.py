import logging
import os
import numpy as np
import torch
from torch import nn
from collections import namedtuple
import time
import datetime
import cv2

from utils import PNGTrainloader, TIFFTrainloader
image_path = "/home/xueruoyao/Documents/PythonProgram/MyFramework/dataset/semantic_segmentation/image"
label_path = "/home/xueruoyao/Documents/PythonProgram/MyFramework/dataset/semantic_segmentation/label"


# def preprocessing(image, label):
#     ret, tem = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
#     return tem, label


# dataiter = PNGIterator(image_path=image_path, label_path=label_path, shuffle=True,
#                        batch_size=50, preprocessing=None, drop_last=True)
# for batch, (images, labels) in enumerate(dataiter):
#     cv2.imshow("dw", 10 * ((np.uint8(np.array(labels)[0]))))
#     cv2.waitKey(0)

a = np.ones((2, 3))
a = a[np.newaxis, :, :]
print(a.shape)