import cv2
import os
import math
import numpy as np
import time
import shutil
from tqdm import tqdm
import imagesize
from numba import njit
from torch import nn
import torch


def label_category_find():
    # calculate all the categories of label
    labels = []
    count = 0
    for label_name in os.listdir("../dataset/semantic_segmentation/label"):
        tem = (cv2.imread(os.path.join("../dataset/semantic_segmentation/label", label_name)))
        labels += list(np.unique(tem[:, :, 2]))
        count += 1
        print(count)
    print(np.unique(np.array(labels)))

@njit
def label_statistics_help(mask, counts: list):
    height, width = mask.shape
    for i in range(height):
        for j in range(width):
            counts[mask[i, j]] += 1
    return counts

def find_damaged_label():
    """ to find if there are damaged gt """
    gt_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/data_aug/gt"
    criterion = nn.CrossEntropyLoss()
    with tqdm(total=len(os.listdir(gt_path))) as pbar:
        for gt_name in os.listdir(gt_path):
            gt = cv2.imread(os.path.join(gt_path, gt_name))[:, :, 2]
            gt = torch.from_numpy(gt).float()
            loss = criterion(gt, gt)
            if(torch.isnan(loss)):
                print("nan: {}".format(gt_name))
            pbar.update()


def label_statistics():
    """ compute each label has how many pixels, labels should be [0, 1, 2...] """
    print("label statistics")
    gt_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/data_aug/gt"
    label = [0, 1]
    counts = np.zeros(len(label))
    with tqdm(total=len(os.listdir(gt_path))) as pbar:
        for gt_name in os.listdir(gt_path):
            mask = cv2.imread(os.path.join(gt_path, gt_name))[:, :, 0]
            counts = label_statistics_help(mask, counts)
            pbar.update()
    sum = 0
    for i in range(len(label)):
        print(f"{i}: {int(counts[i])}")
        sum += int(counts[i])
    print("percentage:")
    for i in range(len(label)):
        print(f"{i}: {round(float(counts[i])/sum, 4)}")


def compute_rgb_mean_std():
    """ calculate mean and std of R, G, B channel"""
    image_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/data_aug/image"

    R_mean, R_std = [], []
    G_mean, G_std = [], []
    B_mean, B_std = [], []
    size = []  # pixels of every image
    sum_size = 0  # 记录所有图片总像素
    with tqdm(total=len(os.listdir(image_path)), unit=" img") as pbar:
        for image_name in os.listdir(image_path):
            rgb_img = cv2.imread(os.path.join(image_path, image_name))
            B, G, R = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
            size.append(len(B))
            sum_size = sum_size + len(B)
            R_mean.append(np.mean(R))
            R_std.append(np.std(R))
            G_mean.append(np.mean(G))
            G_std.append(np.std(G))
            B_mean.append(np.mean(B))
            B_std.append(np.std(B))
            pbar.update()

    num = len(os.listdir(image_path))
    # R_mean
    tem = 0
    for i in range(num):
        tem = tem + size[i] * R_mean[i]
    R_mean_final = tem / sum_size
    print("R_mean:", R_mean_final)

    # R_std
    tem = 0
    for i in range(num):
        tem = tem + size[i] * (math.pow(R_std[i], 2) + math.pow((R_mean_final - R_mean[i]), 2))
    R_std_final = math.sqrt(tem / sum_size)
    print("R_std:", R_std_final)

    # G_mean
    tem = 0
    for i in range(num):
        tem = tem + size[i] * G_mean[i]
    G_mean_final = tem / sum_size
    print("G_mean:", G_mean_final)

    # G_std
    tem = 0
    for i in range(num):
        tem = tem + size[i] * (math.pow(G_std[i], 2) + math.pow((G_mean_final - G_mean[i]), 2))
    G_std_final = math.sqrt(tem / sum_size)
    print("G_std:", G_std_final)

    # B_mean
    tem = 0
    for i in range(num):
        tem = tem + size[i] * B_mean[i]
    B_mean_final = tem / sum_size
    print("B_mean:", B_mean_final)

    # B_std
    tem = 0
    for i in range(num):
        tem = tem + size[i] * (math.pow(B_std[i], 2) + math.pow((B_mean_final - B_mean[i]), 2))
    B_std_final = math.sqrt(tem / sum_size)
    print("B_std:", B_std_final)


def data_clean():
    """ we put messy data into image and gt, perform label transform, with same name """
    data_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/clean"
    save_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/origin"
    if not os.path.exists(os.path.join(save_path)):
        os.makedirs(os.path.join(save_path, "image"))
        os.makedirs(os.path.join(save_path, "gt"))
    assert len(os.listdir(os.path.join(save_path, "image"))) == 0 and \
           len(os.listdir(os.path.join(save_path, "gt"))) == 0, \
           "data clean save path train not empty"
    print("data clean")
    image_name_list = os.listdir(os.path.join(data_path, "image"))
    # for name in os.listdir(data_path):
    #     if name.split(".")[-1] == "jpg":
    #         image_name_list.append(name)

    with tqdm(total=len(image_name_list), unit=" img") as pbar:
        for image_name in image_name_list:
            clean_name = image_name.split("_")[0]
            image = cv2.imread(os.path.join(data_path, "image", image_name))
            cv2.imwrite(os.path.join(save_path, "image", image_name), image)
            gt = cv2.imread(os.path.join(data_path, "gt", image_name))
            gt[gt == 255] = 1
            cv2.imwrite(os.path.join(save_path, "gt", image_name), gt)
            pbar.update()

    print("data: image sum = {}, gt sum = {}".format(
        len(os.listdir(os.path.join(save_path, "image"))),
        len(os.listdir(os.path.join(save_path, "gt")))
    ))


def data_split():
    """ split data into train, valid and test, all has the form of
    image and gt. Image and gt has same name """
    image_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/data_aug/image"
    gt_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/data_aug/gt"
    save_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/new"
    split_ratio = (0.6, 0.2, 0.2)  # train, valid, test
    image_name_list = os.listdir(image_path)
    assert sum(split_ratio) == 1, f"expect split ratio sum = 1, got {sum(split_ratio)}"

    # prepare directories for different data utility
    if split_ratio[0] != 0:
        if not os.path.exists(os.path.join(save_path, "train")):
            os.makedirs(os.path.join(save_path, "train", "image"))
            os.makedirs(os.path.join(save_path, "train", "gt"))
        assert len(os.listdir(os.path.join(save_path, "train", "image"))) == 0 and \
               len(os.listdir(os.path.join(save_path, "train", "gt"))) == 0,\
               "data split save path train not empty"
    if split_ratio[1] != 0:
        if not os.path.exists(os.path.join(save_path, "valid")):
            os.makedirs(os.path.join(save_path, "valid", "image"))
            os.makedirs(os.path.join(save_path, "valid", "gt"))
        assert len(os.listdir(os.path.join(save_path, "valid", "image"))) == 0 and \
               len(os.listdir(os.path.join(save_path, "valid", "gt"))) == 0, \
               "data split save path valid not empty"
    if split_ratio[2] != 0:
        if not os.path.exists(os.path.join(save_path, "test")):
            os.makedirs(os.path.join(save_path, "test", "image"))
            os.makedirs(os.path.join(save_path, "test", "gt"))
        assert len(os.listdir(os.path.join(save_path, "test", "image"))) == 0 and \
               len(os.listdir(os.path.join(save_path, "test", "gt"))) == 0, \
               "data split save path test not empty"

    print("data split")
    tem_train = int(split_ratio[0] * len(image_name_list))
    tem_valid = tem_train + int(split_ratio[1] * len(image_name_list))
    with tqdm(total=len(image_name_list), unit=" img") as pbar:
        # train data
        for image_name in image_name_list[:tem_train]:
            shutil.copy2(os.path.join(image_path, image_name),
                         os.path.join(save_path, "train", "image", image_name))
            shutil.copy2(os.path.join(gt_path, image_name),
                         os.path.join(save_path, "train", "gt", image_name))
            pbar.update()

        # valid data
        for image_name in image_name_list[tem_train:tem_valid]:
            shutil.copy2(os.path.join(image_path, image_name),
                         os.path.join(save_path, "valid", "image", image_name))
            shutil.copy2(os.path.join(gt_path, image_name),
                         os.path.join(save_path, "valid", "gt", image_name))
            pbar.update()

        # test data
        for image_name in image_name_list[tem_valid:]:
            shutil.copy2(os.path.join(image_path, image_name),
                         os.path.join(save_path, "test", "image", image_name))
            shutil.copy2(os.path.join(gt_path, image_name),
                         os.path.join(save_path, "test", "gt", image_name))
            pbar.update()
    print("train: {}, valid: {}, test: {}".format(
        len(os.listdir(os.path.join(save_path, "train", "image"))) if split_ratio[0] != 0 else 0,
        len(os.listdir(os.path.join(save_path, "valid", "image"))) if split_ratio[1] != 0 else 0,
        len(os.listdir(os.path.join(save_path, "test", "image"))) if split_ratio[2] != 0 else 0
    ))


def statistic_image_size():
    """ count unique image size """
    image_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/data_aug/image"
    image_size_dict: dict = {}
    print("statistics of image size")
    with tqdm(total=len(os.listdir(image_path)), unit=" img") as pbar:
        for image_name in os.listdir(image_path):
            width, height = imagesize.get(os.path.join(image_path, image_name))
            if (height, width) not in image_size_dict.keys():
                image_size_dict[(height, width)] = 1
            else:
                image_size_dict[(height, width)] += 1
            pbar.update()
    print(image_size_dict)


if __name__ == "__main__":
    # data_clean()
    # data_split()
    statistic_image_size()
    # compute_rgb_mean_std()
    # label_statistics()
    # find_damaged_label()