import cv2
import os
import numpy as np
import skimage
import imutils
import torch
import tensorflow as tf
from typing import Tuple, Union
import logging


def rotate(image, label, angle):
    """ rotate image and label clockwise """
    image_rotate = imutils.rotate_bound(image, angle)
    label_rotate = imutils.rotate_bound(label, angle)
    return image_rotate, label_rotate


def flip(image, label, flip_code: int):
    """ Flip image and label simultaneously
    :param image: input image
    :param label: corresponding image
    :param flip_code: 1--horizontal, 0--vertical, -1--horizontal and vertical
    """
    image_flip = cv2.flip(image, flip_code)
    label_flip = cv2.flip(label, flip_code)
    return image_flip, label_flip


def resize(image, label, zoom_scale: float):
    """ Resize image and label due to zoom_scale """
    new_height, new_width = int(zoom_scale * image.shape[:2])
    image_out = cv2.resize(image, (new_width, new_height))
    label_out = cv2.resize(label, (new_width, new_height), cv2.INTER_NEAREST)
    return image_out, label_out


def add_noise(image):
    """ add noise to images, Gaussian noise by default """
    out = skimage.util.random_noise(image, mode='gaussian', var=0.01)
    return np.uint8(out * 256)


def random_variation(image, mode: int):
    """ whole image variation
    :param image:input image
    :param mode: 0--random_brightness, 1--random_contrast, 2--random_hue, 3--random_saturation
    :return: image after processing
    """
    if mode == 0:
        random_brightness = tf.image.random_brightness(image, max_delta=0.25)
        return np.uint8(random_brightness)
    elif mode == 1:
        random_contrast = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        return np.uint8(random_contrast)
    elif mode == 2:
        random_hue = tf.image.random_hue(image, max_delta=0.3)
        return np.uint8(random_hue)
    elif mode == 3:
        random_satu = tf.image.random_saturation(image, lower=0.2, upper=1.8)
        return np.uint8(random_satu)


def random_crop(image, label, chip_size: Union[int, Tuple[int, int]]):
    """ random crop of both image and label, chipped size (chip_height, chip_width) or int """
    if isinstance(chip_size, int):
        chip_height, chip_width = chip_size, chip_size
    else:
        chip_height, chip_width = chip_size
    height, width = image.shape[:2]
    x = np.random.randint(0, height - chip_height)
    y = np.random.randint(0, width - chip_width)
    chip_image = image[x:x + chip_height, y:y + chip_width]
    chip_label = label[x:x + chip_height, y:y + chip_width]
    return chip_image, chip_label


def random_mosaic(image_list, label_list, size: int):
    """
    take four images and corresponding labels. Random crop, rotate and flip,
    finally stitch together to a single image and label

    :param image_list: 4 images to crop
    :param label_list: 4 corresponding labels
    :param size: output size will be (size, size)
    :return:mosaic image
    """
    channel = image_list[0].shape[2]
    image_out = np.uint8(np.zeros([size, size, channel]))
    label_out = np.uint8(np.zeros([size, size, channel]))

    # set the size of upper left image to be (a,b), ensure four chipped images not too small
    a = np.random.randint(int(size * 0.3), int(size * 0.7))
    b = np.random.randint(int(size * 0.3), int(size * 0.7))
    for i in torch.randperm(4):
        image = image_list[i]
        label = label_list[i]

        # random crop
        if i == 0:
            chip_image, chip_label = random_crop(image, label, (a, b))
        elif i == 1:
            chip_image, chip_label = random_crop(image, label, (a, size - b))
        elif i == 2:
            chip_image, chip_label = random_crop(image, label, (size - a, b))
        elif i == 3:
            chip_image, chip_label = random_crop(image, label, (size - a, size - b))

        # random rotate and flip
        random_angle = np.random.choice([0, 180])
        random_flip = np.random.choice([-1, 0, 1])
        chip_image, chip_label = rotate(chip_image, chip_label, random_angle)
        chip_image, chip_label = flip(chip_image, chip_label, random_flip)

        # image stitching
        if i == 0:
            image_out[0:a, 0:b, :] = chip_image
            label_out[0:a, 0:b] = chip_label
        elif i == 1:
            image_out[0:a, b:size, :] = chip_image
            label_out[0:a, b:size] = chip_label
        elif i == 2:
            image_out[a:size, 0:b, :] = chip_image
            label_out[a:size, 0:b] = chip_label
        elif i == 3:
            image_out[a:size, b:size, :] = chip_image
            label_out[a:size, b:size] = chip_label

    return image_out, label_out


def crop_and_resize(image, label, crop_size, resize_size, overlap_ratio, save_path, count):
    """ to crop a large image into small images and resize
    :param image:
    :param label:
    :param crop_size: input int a, then shape(int, int)
    :param resize_size: the size image resize to
    :param overlap_ratio: it decides the stride for crop
    :param save_path: where to save images and labels
    :param count: saved images count, in order to give different names to images
    :return:
    """   
    if not os.path.exists(os.path.join(save_path, "image")):
        os.makedirs(os.path.join(save_path, "image"))
    if not os.path.exists(os.path.join(save_path, "label")):
        os.makedirs(os.path.join(save_path, "label"))
    save_image_path = os.path.join(save_path, "image")
    save_label_path = os.path.join(save_path, "label")

    img_height, img_width = image.shape[:2]
    if img_height <= crop_size or img_width <= crop_size:
        return count

    stride = int(crop_size * overlap_ratio)
    for x in range(0, img_width, stride - crop_size):
        for y in range(0, img_height, stride - crop_size):
            # (x, y) is the lower right coordinate of the image
            crop_image = image[y - crop_size: y, x - crop_size: x, :]
            crop_label = label[y - crop_size: y, x - crop_size: x, :]
            if crop_size != resize_size:
                crop_image = cv2.resize(crop_image, (resize_size, resize_size))
                crop_label = cv2.resize(crop_label, (resize_size, resize_size), cv2.INTER_NEAREST)
            count += 1
            cv2.imwrite(os.path.join(save_image_path, str(count) + ".png"), crop_image)
            cv2.imwrite(os.path.join(save_label_path, str(count) + ".png"), crop_label)
    return count


if __name__ == "__main__":
    image_path = "/home/xueruoyao/Documents/PythonProgram/MyFramework/dataset/semantic_segmentation/original/image"
    label_path = "/home/xueruoyao/Documents/PythonProgram/MyFramework/dataset/semantic_segmentation/original/gt"
    save_path = "/home/xueruoyao/Documents/PythonProgram/MyFramework/dataset/semantic_segmentation/data_aug"
    if not os.path.exists(os.path.join(save_path, "image")):
        os.makedirs(os.path.join(save_path, "image"))
    if not os.path.exists(os.path.join(save_path, "gt")):
        os.makedirs(os.path.join(save_path, "gt"))

    num = 0  # image saved count
    mosaic_image_list = []  # images for random mosaic
    mosaic_label_list = []  # labels for random mosaic

    for image_name in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, image_name))
        gt = cv2.imread(os.path.join(label_path, image_name))
        mosaic_image_list.append(img)
        mosaic_label_list.append(gt)

        if len(mosaic_image_list) == 4:
            num += 1
            image_mosaic, label_mosaic = random_mosaic(mosaic_image_list, mosaic_label_list, 512)
            mosaic_image_list.clear()
            mosaic_label_list.clear()
            cv2.imwrite(os.path.join(save_path, "image", "{}.png".format(num)), image_mosaic)
            cv2.imwrite(os.path.join(save_path, "gt", "{}.png".format(num)), label_mosaic)

        img, gt = random_crop(img, gt, 512)
        random_angle = np.random.choice([0, 90, 180, 270])
        random_flip = np.random.choice([-1, 0, 1])
        img, gt = rotate(img, gt, random_angle)
        img, chip_label = flip(img, gt, random_flip)
        num += 1
        cv2.imwrite(os.path.join(save_path, "image", "{}.png".format(num)), img)
        cv2.imwrite(os.path.join(save_path, "gt", "{}.png".format(num)), gt)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logging.info(f"image saved count = {num}")
