"""
Taking into consideration that pytorch dataset and dataloader
seem to be clumsy for daily self-use, I would like to construct a
specialised light-weight dataloader for training and testing

TODO: need to adjust fetcher to make it more readable
TODO: add num_workers for data load parallelization
"""
import logging
import numpy as np
import torch
import cv2
import os
import datetime
from torchvision import transforms
from typing import List, Callable, Optional, Tuple
import rasterio
import imagesize


class ComputerVisionTestLoader:
    """ Base class for test loader for computer vision
    :param image_path: image path
    :param stride: stride between two chipped images
    :param batch_size: how many samples per batch to load
    :param preprocessing: if not None, use preprocessing function after loading

    Note: we save all chip images information in self.chip_information in form of
          (image_index, height_coord, width_coord)
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
        self.chip_information = []
        self.normalisation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(89.8, 91.3, 89.9), std=(68.0, 65.6, 66.4))
        ])
        self.prepare_chip_information()

    def prepare_chip_information(self):
        """ prepare chip information, saved in self.chip_information """
        if not os.path.exists(self.image_path):
            raise FileNotFoundError("testloader image path not exists")
        elif len(os.listdir(self.image_path)) == 0:
            raise FileNotFoundError("testloader image path is empty")

        for image_name in os.listdir(self.image_path):
            self.image_name_list.append(image_name)
            self.image_path_list.append(os.path.join(self.image_path, image_name))

        # find all possible upper left coordinates for chipped images,
        # save as (image_index, height_coord, width_coord)
        for count in range(len(self.image_path_list)):
            width, height = imagesize.get(self.image_path_list[count])
            assert width >= self.chipsize and height >= self.chipsize,\
                f"chipsize {self.chipsize} doesn't work for " \
                f"{self.image_path_list[count]} with size ({height}, {width})"

            list_height = np.unique(np.array(list(range(0, (height-self.chipsize), self.stride)) +
                                             [height-self.chipsize, 0]))  # height coord
            list_width = np.unique(np.array(list(range(0, (width-self.chipsize), self.stride)) +
                                            [width-self.chipsize, 0]))  # width coord
            for i in list_height:
                for j in list_width:
                    self.chip_information.append((count, i, j))

    def sampler(self):
        """ start_index, end_index of self.chip_info for different batches"""
        for i in range(0, len(self.chip_information), self.batch_size):
            yield i, min(i + self.batch_size, len(self.chip_information))

    def fetcher(self, start_index: int, end_index: int):
        """
        return normalised chipped images of given indices,
        shape (batch_size, channel, height, width)
        """
        chip_info = self.chip_information[start_index: end_index]
        index = chip_info[0][0]  # record current image index (large image)
        image = self.normalisation(self.load(self.image_path_list[index]))  # (channel, height, width)
        distributed_images = []  # record chipped images to be distributed

        # load image and chip, info: (image_index, height_coord, width_coord)
        for info in chip_info:
            if info[0] != index:  # means need to load another image to chip
                index = info[0]
                image = self.normalisation(self.load(self.image_path_list[index]))
            distributed_images.append(
                image[:, info[1]: info[1]+self.chipsize, info[2]: info[2]+self.chipsize])

        return torch.stack(distributed_images, dim=0), torch.tensor(chip_info)

    def stitcher(self, preds: torch.Tensor, info, last_batch_flag: bool = False):
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

        # initialisation for first batch
        if self.current_image_index is None:
            self.current_image_index = info[0, 0]
            width, height = imagesize.get(self.image_path_list[self.current_image_index])
            self.count = np.zeros([height, width])
            self.whole_image = np.zeros([self.n_class, height, width])

        for i in range(info.shape[0]):
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
            print("last image")
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
        return len(self.chip_information)


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

