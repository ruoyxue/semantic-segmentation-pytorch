"""
Taking into consideration that pytorch dataset and dataloader
seem to be clumsy for daily self-use, I would like to construct a
specialised light-weight trainloader for data preparation
"""
import numpy as np
import torch
import cv2
import os
import datetime
from typing import List, Callable, Optional, Tuple
import rasterio


class ComputerVisionTestLoader:
    """ Base class for test loader for computer vision
    :param image_path: image path
    :param label_path: label path
    :param batch_size: how many samples per batch to load
    :param drop_last: if True, drop the last incomplete batch,
    :param shuffle: if True, shuffle data in __iter__
    :param preprocessing: if not None, use preprocessing function after loading

    Note: by default, we set image's name and label's name to be the same,
          it's suggested that you set your own way via method
          `prepare_image_name_list`
    """
    def __init__(self, image_path: str, label_path: str, batch_size: int = 1,
                 drop_last: bool = False, shuffle: bool = False,
                 preprocessing: Optional[Callable] = None):
        self.image_path = image_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.preprocessing = preprocessing
        self.image_path_list = []
        self.label_path_list = []
        self.prepare_image_label_list()

    def prepare_image_label_list(self):
        """ save image's and label's absolute path correspondingly """
        for image_name in os.listdir(self.image_path):
            if os.path.exists(os.path.join(self.label_path, image_name)):
                self.image_path_list.append(os.path.join(self.image_path, image_name))
                self.label_path_list.append(os.path.join(self.label_path, image_name))

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
        return images and labels of given indices, with possible preprocessing,
        in shape (batch_size, channel, height, width)
        """
        images, labels = [], []
        for index in indices:
            image = self.load(self.image_path_list[index], "image")
            label = self.load(self.label_path_list[index], "label")
            if self.preprocessing is not None:
                image, label = self.preprocessing(image, label)
            image = np.rollaxis(image, 2, 0)
            images.append(torch.tensor(image, dtype=torch.float))
            labels.append(torch.tensor(label, dtype=torch.int64))
        return torch.stack(images, dim=0), torch.stack(labels, dim=0)

    def load(self, path: str, mode: str):
        """ load image/label
        :return data: np.array
        Note: image:(B, G, R) (height, width, channel) label: (height, width)
         """
        raise NotImplementedError

    def __iter__(self):
        for indices in self.sampler():
            yield self.fetcher(indices)

    def __len__(self):
        assert len(self.image_path_list) == len(self.label_path_list),\
            "image path list doesn't have the same len as label path list"
        return len(self.image_path_list)


class PNGTestloader(ComputerVisionTestLoader):
    """ subclass to read and solve png files """
    def __init__(self, image_path: str, label_path: str, batch_size: int = 1,
                 drop_last: bool = False, shuffle: bool = False,
                 preprocessing: Optional[Callable] = None):
        super().__init__(image_path, label_path, batch_size, drop_last, shuffle, preprocessing)

    def load(self, path: str, mode: str) -> np.array:
        if mode == "image":
            return cv2.imread(path)
        if mode == "label":
            tem = cv2.imread(path)[:, :, 2]
            return tem


class TIFFTestloader(ComputerVisionTestLoader):
    """ subclass to read and solve tiff tiles """
    def __init__(self, image_path: str, label_path: str, batch_size: int = 1,
                 drop_last: bool = False, shuffle: bool = False,
                 preprocessing: Optional[Callable] = None):
        super().__init__(image_path, label_path, batch_size, drop_last, shuffle, preprocessing)

    def load(self, path: str, mode: str) -> np.array:
        # return data has form of B, G, R
        if mode == "image":
            with rasterio.open(path) as file:
                data = cv2.merge([file.read(3), file.read(2), file.read(1)])
            return data
        if mode == "label":
            with rasterio.open(path) as file:
                data = file.read(1)
                return data

