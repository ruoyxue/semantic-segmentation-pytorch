"""
Taking into consideration that pytorch dataset and dataloader
seem to be clumsy for daily self-use, I would like to construct a
specialised light-weight dataloader for training and testing

TODO: add num_workers for data load parallelization
"""
import numpy as np
import torch
import cv2
import os
import datetime
from typing import List, Callable, Optional, Tuple
import rasterio
import torch.nn as nn
from torchvision import transforms
import preprocessing as prepro


class ComputerVisionTrainLoader:
    """ Base class for train loader for computer vision
    :param image_path: image path
    :param gt_path: label path
    :param batch_size: how many samples per batch to load
    :param drop_last: if True, drop the last incomplete batch,
    :param shuffle: if True, shuffle data in __iter__
    :param chip_size: control random crop in preprocessing

    Note: by default, we set image's name and label's name to be the same,
          it's suggested that you set your own way via method
          `prepare_image_name_list`
    """
    def __init__(self, image_path: str, gt_path: str, batch_size: int = 1,
                 drop_last: bool = False, shuffle: bool = False, chip_size: int = 512):
        self.image_path = image_path
        self.gt_path = gt_path
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.image_path_list = []
        self.gt_path_list = []
        self.preprocessing = prepro.ProcessingSequential([
            prepro.RandomCrop(chip_size=chip_size),
            prepro.RandomRotate(random_choice=[0, 90, 180, 270]),
            prepro.RandomFlip(random_choice=[-1, 0, 1]),
            prepro.Normalize(mean=(73.4711, 97.6228, 104.4753), std=(31.2603, 32.3015, 39.8499)),
            prepro.ToTensor()
        ])
        self.prepare_image_label_list()

    def prepare_image_label_list(self):
        """ save image's and label's absolute path correspondingly """
        if not os.path.exists(self.image_path) or not os.path.exists(self.gt_path):
            raise FileNotFoundError("trainloader image path or gt path not exists")
        elif len(os.listdir(self.image_path)) == 0 or len(os.listdir(self.gt_path)) == 0:
            raise FileNotFoundError("trainloader image path or gt path is empty")

        for image_name in os.listdir(self.image_path):
            if os.path.exists(os.path.join(self.gt_path, image_name)):
                self.image_path_list.append(os.path.join(self.image_path, image_name))
                self.gt_path_list.append(os.path.join(self.gt_path, image_name))
        assert len(self.image_path) > 0, \
            "trainloader can't find images and labels to distribute, they must enjoy the same name"

    def sampler(self):
        """ yield indices of each batch """
        indices = torch.tensor(range(len(self)))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(int((datetime.datetime.now().strftime("%Y%m%d%H%M%S"))))
            indices = torch.randperm(len(self), generator=generator)
        if self.drop_last and (len(self) % self.batch_size) != 0:
            indices = indices[:-(len(self) % self.batch_size)]
        for i in range(0, len(indices), self.batch_size):
            yield indices[i: min(i + self.batch_size, len(self))]

    def fetcher(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return images and labels of given indices, with preprocessing,
        in shape (batch_size, channel, height, width)
        """
        images, labels = [], []
        for index in indices:
            image = self.load(self.image_path_list[index], "image")
            gt = self.load(self.gt_path_list[index], "gt")
            image, gt = self.preprocessing(image, gt)
            images.append(image.permute([2, 0, 1]))
            labels.append(gt)
        return torch.stack(images, dim=0), torch.stack(labels, dim=0)

    def load(self, path: str, mode: str):
        """ load image/label
        :return data: np.array
        image: (height, width, channel) (B, G, R)
        label: (height, width)
         """
        raise NotImplementedError

    def __iter__(self):
        for indices in self.sampler():
            yield self.fetcher(indices)

    def __len__(self):
        assert len(self.image_path_list) == len(self.gt_path_list),\
            "image path list doesn't have the same len as label path list"
        return len(self.image_path_list)

    def state_dict(self):
        return {
            "trainloader_type": str(type(self)),
            "drop_last": self.drop_last,
            "shuffle": self.shuffle,
            "preprocessing": self.preprocessing.list_of_repr()
        }


class PNGTrainloader(ComputerVisionTrainLoader):
    """ subclass to read and solve png files """
    def __init__(self, image_path: str, gt_path: str, batch_size: int = 1,
                 drop_last: bool = False, shuffle: bool = False, chip_size: int = 512):
        super().__init__(image_path, gt_path, batch_size, drop_last, shuffle, chip_size)

    def load(self, path: str, mode: str) -> np.array:
        if mode == "image":
            return cv2.imread(path)
        if mode == "gt":
            return cv2.imread(path)[:, :, 2]


class TIFFTrainloader(ComputerVisionTrainLoader):
    """ subclass to read and solve tiff tiles """
    def __init__(self, image_path: str, gt_path: str, batch_size: int = 1,
                 drop_last: bool = False, shuffle: bool = False, chip_size: int = 512):
        super().__init__(image_path, gt_path, batch_size, drop_last, shuffle, chip_size)

    def load(self, path: str, mode: str) -> np.array:
        # return data has form of B, G, R
        if mode == "image":
            with rasterio.open(path) as data:
                return cv2.merge([data.read(3), data.read(2), data.read(1)])
        if mode == "gt":
            with rasterio.open(path) as data:
                return data.read(1)
