"""
Dynamic data augmentation and other preprocessing methods
"""

import cv2
import numpy as np
import skimage
import imutils
import torch
import tensorflow as tf
from typing import Tuple, Union, List, Optional


class ProcessingSequential:
    """ Sequential of preprocessing methods """
    def __init__(self, sequence: List):
        self.sequence = sequence

    def __call__(self, image: np.array, gt: Optional[np.array] = None):
        if gt is None:
            gt = np.zeros_like(image)
        for processing in self.sequence:
            image, gt = processing(image, gt)
        return image, gt

    def __getitem__(self, item: int):
        if not isinstance(item, int):
            raise RuntimeError("Preprocessing ProcessingSequential needs int as index, not string")
        return self.sequence[item]

    def __repr__(self):
        output_string = ""
        for processing in self.sequence:
            output_string += processing.__class__.__name__ + "("
            for key in processing.__dict__.keys():
                output_string += key + "=" + str(processing.__dict__[key]) + ", "
            output_string = output_string.strip(", ") + ")\n"
        return output_string.strip("\n")

    def list_of_repr(self):
        """ store str of each preprocessing method in a list and return """
        output_list = []
        for processing in self.sequence:
            pro_str = processing.__class__.__name__ + "("
            for key in processing.__dict__.keys():
                pro_str += key + "=" + str(processing.__dict__[key]) + ", "
            pro_str = pro_str.strip(", ") + ")"
            output_list.append(pro_str)
        return output_list


class ToTensor:
    def __call__(self, image: np.array, gt: np.array):
        return torch.from_numpy(image).float(), torch.from_numpy(gt)


class Normalize:
    def __init__(self, mean: Tuple, std: Tuple):
        assert len(mean) == len(std), \
            f"Preprocessing Normalise expects same-size mean and std, got {len(mean)} and {len(std)}"
        self.mean = mean
        self.std = std

    def __call__(self, image: np.array, gt: np.array):
        assert len(self.mean) == image.shape[2], \
            f"Preprocessing Normalise len(mean) must equal to ({len(self.mean)}) and image.shape[2]({image.shape[2]})"
        image = np.float32(image)
        gt = np.int64(gt)
        for i in range(len(self.mean)):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        return image, gt


class RandomRotate:
    """ rotate image and gt clockwise randomly
    :param random_choice: it can randomly choose angle in this list
    """
    def __init__(self, random_choice: List[int]):
        self.random_choice = random_choice

    def __call__(self, image: np.array, gt: np.array):
        angle = np.random.choice(self.random_choice)
        return imutils.rotate_bound(image, angle), imutils.rotate_bound(gt, angle)


class RandomFlip:
    """ flip image and gt randomly
    :param random_choice(List): 1--horizontal, 0--vertical, -1--horizontal and vertical
    """
    def __init__(self, random_choice: List[int]):
        for i in random_choice:
            assert i in [-1, 0, 1], \
                f"Augmentation RandomFlip expects random_choice in [-1, 0, 1], got {i}"
        self.random_choice = random_choice

    def __call__(self, image: np.array, gt: np.array):
        flip_code = np.random.choice(self.random_choice)
        return cv2.flip(image, flip_code), cv2.flip(gt, flip_code)


class ZoomScaleResize:
    """ resize image and gt in terms of zoom scale """
    def __init__(self, zoom_scale: float, mode: int = cv2.INTER_NEAREST):
        assert zoom_scale > 0, f"Augmentation ResizeScale expects zoom_scale > 0, got {zoom_scale}"
        self.zoom_scale = zoom_scale
        self.mode = mode

    def __call__(self, image: np.array, gt: np.array):
        new_height, new_width = int(self.zoom_scale * image.shape[:2])
        return cv2.resize(image, (new_width, new_height), self.mode),\
            cv2.resize(gt, (new_width, new_height), self.mode)


class FixedSizeResize:
    """ resize image and gt to a fixed size
    :param new_shape: (new_width, new_height)
    """
    def __init__(self, new_shape: Tuple[int, int], mode: int = cv2.INTER_NEAREST):
        self.new_shape = new_shape
        self.mode = mode

    def __call__(self, image: np.array, gt: np.array):
        return cv2.resize(image, self.new_shape, self.mode),\
               cv2.resize(gt, self.new_shape, self.mode)


class RandomCrop:
    """ random crop of both image and label
    :param chip_size: (chip_height, chip_width) or int
    """
    def __init__(self, chip_size: Union[int, Tuple[int, int]]):
        if isinstance(chip_size, int):
            self.chip_height, self.chip_width = chip_size, chip_size
        else:
            self.chip_height, self.chip_width = chip_size

    def __call__(self, image: np.array, gt: np.array):
        height, width = image.shape[:2]
        if self.chip_height == height and self.chip_width == width:
            return image, gt
        x = np.random.randint(0, height - self.chip_height)
        y = np.random.randint(0, width - self.chip_width)
        return image[x:x + self.chip_height, y:y + self.chip_width], \
            gt[x:x + self.chip_height, y:y + self.chip_width]


class AddNoise:
    """ add noise to image using skimage.util.random_noise """
    def __init__(self, mode: str = "gaussian", var: float = 0.01):
        assert var > 0, f"Augmentation AddNoise expects var > 0, got {var}"
        self.mode = mode
        self.var = var

    def __call__(self, image: np.array, gt: np.array):
        return np.uint8(skimage.util.random_noise(image, self.mode, self.var) * 256), gt


class RandomVariation:
    """ image variation
    :param mode: 0--random_brightness, 1--random_contrast, 2--random_hue, 3--random_saturation
    """
    def __init__(self, mode: str):
        assert mode in [0, 1, 2, 3],\
            f"Augmentation RandomVariation expects mode in [0, 1, 2, 3], got {mode}"
        self.mode = mode

    def __call__(self, image: np.array, gt: np.array):
        if self.mode == 0:
            return np.uint8(tf.image.random_brightness(image, max_delta=0.25))
        elif self.mode == 1:
            return np.uint8(tf.image.random_contrast(image, lower=0.2, upper=1.8))
        elif self.mode == 2:
            return np.uint8(tf.image.random_hue(image, max_delta=0.3))
        elif self.mode == 3:
            return np.uint8(tf.image.random_saturation(image, lower=0.2, upper=1.8))


class RandomMosaic:
    """ random mosaic augmentation
    This augmentation is different from above, as it needs 4 images and gts per call

    take 4 images and corresponding gts. Random crop, flip, finally stitch together to a single image and gt
    :param final_size: output would be (int, int)
    :param n_channel: channel of image, 3 for jpg png, 4 for tiff
    """
    def __init__(self, final_size: int, n_channel: int):
        self.final_size = final_size
        self.n_channel = n_channel
        self.random_flip = RandomFlip([-1, 0, 1])

    @staticmethod
    def random_crop(image: np.array, gt: np.array, crop_size: Tuple[int, int]):
        height, width = image.shape[:2]
        chip_height, chip_width = crop_size
        x = np.random.randint(0, height - chip_height)
        y = np.random.randint(0, width - chip_width)
        return image[x:x + chip_height, y:y + chip_width], \
            gt[x:x + chip_height, y:y + chip_width]

    def __call__(self, image_list: List[np.array], gt_list: List[np.array]):
        image_out = np.uint8(np.zeros([self.final_size, self.final_size, self.n_channel]))
        label_out = np.uint8(np.zeros([self.final_size, self.final_size]))
        # set the size of upper left image to be (a,b), ensure four chipped images not too small
        a = np.random.randint(int(self.final_size * 0.3), int(self.final_size * 0.7))
        b = np.random.randint(int(self.final_size * 0.3), int(self.final_size * 0.7))
        a_b_list = [(a, b), (a, self.final_size - b), (self.final_size - a, b),
                    (self.final_size - a, self.final_size - b)]

        for i in torch.randperm(4):
            image = image_list[i]
            gt = gt_list[i]
            assert image.ndim == 3 and gt.ndim == 2, \
                f"Augmentation RandomMosaic expects image.ndim = 3 and gt.ndim = 2, got {image.ndim} and {gt.ndim}"
            assert image.shape[2] == self.n_channel, "Augmentation RandomMosaic got image.shape[2] != self.n_channel"

            # random crop
            chip_image, chip_gt = RandomMosaic.random_crop(image, gt, a_b_list[i])

            # random rotate and flip
            chip_image, chip_gt = self.random_flip(chip_image, chip_gt)
            chip_image, chip_gt = self.random_flip(chip_image, chip_gt)

            # image stitching
            if i == 0:
                image_out[0:a, 0:b, :] = chip_image
                label_out[0:a, 0:b] = chip_gt
            elif i == 1:
                image_out[0:a, b:self.final_size, :] = chip_image
                label_out[0:a, b:self.final_size] = chip_gt
            elif i == 2:
                image_out[a:self.final_size, 0:b, :] = chip_image
                label_out[a:self.final_size, 0:b] = chip_gt
            elif i == 3:
                image_out[a:self.final_size, b:self.final_size, :] = chip_image
                label_out[a:self.final_size, b:self.final_size] = chip_gt

        return image_out, label_out
