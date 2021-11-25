"""
Taking into consideration that pytorch dataset and dataloader
seem to be clumsy for daily self-use, I would like to construct a
specialised light-weight trainloader for data preparation

:TODO need to adjust fetcher to make it more readable
"""
import logging

import numpy as np
import torch
import cv2
import os
import datetime
from typing import List, Callable, Optional, Tuple
import rasterio
import imagesize


class ComputerVisionTestLoader:
    """ Base class for test loader for computer vision
    :param image_path: image path
    :param stride: stride between two chipped images
    :param batch_size: how many samples per batch to load
    :param preprocessing: if not None, use preprocessing function after loading

    Note: we save all chip images information in self.chip_info in form of
          (image_index, height_coord, width_coord),
          where image_index refers to the image that chipped image belongs to (in terms of os.listdir)
          , (height_coord, width_coord) denotes the upper-left point coordinate of chipped image
    """
    whole_image: torch.tensor = None
    count: torch.tensor = None
    current_image_index: int = None

    def __init__(self, image_path: str, chip_size: int, stride: int, n_class: int,
                 batch_size: int = 1, preprocessing: Optional[Callable] = None):
        self.image_name_list = []
        self.image_path_list = []
        self.image_path = image_path
        self.chipsize = chip_size
        self.stride = stride
        self.n_class = n_class
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.chip_info = []
        self.prepare_chip_info()

    def prepare_chip_info(self):
        """ prepare chip_info """
        if not os.path.exists(self.image_path):
            raise FileNotFoundError("testloader image path not exists")
        elif len(os.listdir(self.image_path)) == 0:
            raise FileNotFoundError("testloader image path is empty")

        for image_name in os.listdir(self.image_path):
            self.image_name_list.append(image_name)
            self.image_path_list.append(os.path.join(self.image_path, image_name))

        for count in range(len(self.image_path_list)):
            width, height = imagesize.get(self.image_path_list[count])
            assert width >= self.chipsize and height >= self.chipsize,\
                f"chipsize doesn't work for {self.image_path_list[count]} with size of ({height}, {width})"

            list_i = np.unique(np.array(list(range(0, (height - self.chipsize), self.stride)) +
                                        [height - self.chipsize, 0]))
            list_j = np.unique(np.array(list(range(0, (width - self.chipsize), self.stride)) +
                                        [width - self.chipsize, 0]))
            for i in list_i:
                for j in list_j:
                    self.chip_info.append((count, i, j))

    def sampler(self):
        """ start_index, end_index for chip images in self.chip_info """
        for i in range(0, len(self), self.batch_size):
            yield i, min(i + self.batch_size, len(self))

    def fetcher(self, start_index: int, end_index: int):
        """
        return chipped images of given indices, with possible preprocessing,
        in shape (batch_size, channel, height, width)
        """
        chipped_images, info = [], []
        chipped_info = self.chip_info[start_index: end_index]
        # split chipped_info in terms of their whole image index
        image_indices = np.unique(np.array(chipped_info)[:, 0])  # get whole image index
        for image_index in image_indices:
            info.append([])
        for i in range(self.batch_size):
            info[chipped_info[i][0] - min(image_indices)].append(chipped_info[i])
        # load whole image and chip
        for element in info:
            image = self.load(self.image_path_list[element[0][0]])
            if self.preprocessing is not None:
                image = self.preprocessing(image)
            image = np.rollaxis(image, 2, 0)
            for index, height_coord, width_coord in element:
                chipped_images.append(torch.tensor(
                      image[:, height_coord: height_coord + self.chipsize, width_coord: width_coord + self.chipsize],
                      dtype=torch.float))
        return torch.stack(chipped_images, dim=0), np.array(chipped_info)

    def stitcher(self, preds, info, last_batch_flag: bool = False):
        """ stitch the preds together and return whole predicted image and its name
        :param preds: predictions of chipped images
        :param info: information of chipped images
        :param last_batch_flag: if this is the last batch
        """
        # adding higher weights for pixels which are in the center, in order to mitigate edge effects
        preds = np.array(preds.detach().cpu())
        info = np.array(info)
        half_stride = self.stride // 2
        kernel = np.ones((self.chipsize, self.chipsize), dtype=np.float32)
        kernel[half_stride:-half_stride, half_stride:-half_stride] = 10
        for i in range(info.shape[0]):
            # initialisation for first batch
            if self.current_image_index is None:
                self.current_image_index = info[0, 0]
                width, height = imagesize.get(self.image_path_list[self.current_image_index])
                self.count = np.zeros([height, width])
                self.whole_image = np.zeros([self.n_class, height, width])

            if info[i, 0] == self.current_image_index:
                # add chipped image to whole_image
                self.whole_image[:, info[i, 1]: info[i, 1]+self.chipsize, info[i, 2]: info[i, 2]+self.chipsize] += \
                    preds[i] * kernel
                self.count[info[i, 1]: info[i, 1]+self.chipsize, info[i, 2]: info[i, 2]+self.chipsize] += kernel
            else:
                self.whole_image /= self.count
                # return whole image that after stitching
                yield self.whole_image, self.image_name_list[self.current_image_index]

                # start to stitch new image
                self.current_image_index = info[i, 0]
                width, height = imagesize.get(self.image_path_list[self.current_image_index])
                self.count = torch.zeros([height, width])
                self.whole_image = torch.zeros([self.n_class, height, width])
                self.whole_image[:, info[i, 1]: info[i, 1] + self.chipsize, info[i, 2]: info[i, 2] + self.chipsize] += \
                    preds[i] * kernel
                self.count[info[i, 1]: info[i, 1] + self.chipsize, info[i, 2]: info[i, 2] + self.chipsize] += kernel

        if last_batch_flag is True:
            self.whole_image /= self.count
            yield self.whole_image, self.image_name_list[self.current_image_index]

    def load(self, path: str):
        """ load image
        Note: image (B, G, R), (height, width, channel), np.array
         """
        raise NotImplementedError

    def __iter__(self):
        for start_index, end_index in self.sampler():
            yield self.fetcher(start_index, end_index)

    def __len__(self):
        return len(self.chip_info)


class PNGTestloader(ComputerVisionTestLoader):
    """ subclass to read and solve png files """

    def __init__(self, image_path: str, chip_size: int, stride: int, n_class: int,
                 batch_size: int = 1, preprocessing: Optional[Callable] = None):
        super().__init__(image_path, chip_size, stride, n_class, batch_size, preprocessing)

    def load(self, path: str) -> np.array:
        # return data has form of B, G, R
        return cv2.imread(path)


class TIFFTestloader(ComputerVisionTestLoader):
    """ subclass to read and solve tiff tiles """
    def __init__(self, image_path: str, chip_size: int, stride: int, n_class: int,
                 batch_size: int = 1, preprocessing: Optional[Callable] = None):
        super().__init__(image_path, chip_size, stride, n_class, batch_size, preprocessing)

    def load(self, path: str) -> np.array:
        # return data has form of B, G, R
        with rasterio.open(path) as file:
            return cv2.merge([file.read(3), file.read(2), file.read(1)])

