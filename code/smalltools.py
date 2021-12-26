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
import warnings
import torch
import pandas as pd


def label_category_find():
    # calculate all the categories of label
    gt_path = "../dataset/semantic_segmentation/label"
    print("label_category_find")
    labels = []
    count = 0
    for label_name in os.listdir(gt_path):
        tem = (cv2.imread(os.path.join(gt_path, label_name)))
        labels += list(np.unique(tem[:, :, 2]))
        count += 1
        print(count)
    print(np.unique(np.array(labels)))


def find_damaged_label():
    """ to find if there are damaged gt """
    print("find_damaged_label")
    # gt_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/data_aug/gt"
    gt_path = "/home/xueruoyao/Documents/PythonProgram/dataset/data_aug/gt"
    criterion = nn.CrossEntropyLoss()
    with tqdm(total=len(os.listdir(gt_path))) as pbar:
        for gt_name in os.listdir(gt_path):
            gt = cv2.imread(os.path.join(gt_path, gt_name))[:, :, 2]
            gt = torch.from_numpy(gt).float()
            loss = criterion(gt, gt)
            if torch.isnan(loss):
                print("nan: {}".format(gt_name))
            pbar.update()


@njit
def label_statistics_help(mask, counts: list):
    height, width = mask.shape
    for i in range(height):
        for j in range(width):
            counts[mask[i, j]] += 1
    return counts


def label_statistics():
    """ compute each label has how many pixels, labels should be [0, 1, 2...] """
    print("label statistics")
    gt_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/segmented/train/gt"
    label = [0, 1]
    counts = np.zeros(len(label))
    with tqdm(total=len(os.listdir(gt_path))) as pbar:
        for gt_name in os.listdir(gt_path):
            mask = cv2.imread(os.path.join(gt_path, gt_name))[:, :, 0]
            counts = label_statistics_help(mask, counts)
            pbar.update()
    sum_count = 0
    for i in range(len(label)):
        print(f"{i}: {int(counts[i])}")
        sum_count += int(counts[i])
    print("percentage:")
    for i in range(len(label)):
        print(f"{i}: {round(float(counts[i])/sum_count, 4)}")


def compute_rgb_mean_std():
    """ calculate mean and std of R, G, B channel """
    print("compute_rgb_mean_std")
    # image_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/data_aug/image"
    image_path = "/home/xueruoyao/Documents/PythonProgram/dataset/deepglobe/image"
    image_name_list = os.listdir(image_path)
    image_count = len(image_name_list)
    r_mean_array, r_std_array = np.zeros(image_count), np.zeros(image_count)
    g_mean_array, g_std_array = np.zeros(image_count), np.zeros(image_count)
    b_mean_array, b_std_array = np.zeros(image_count), np.zeros(image_count)

    single_pixel_count = np.zeros(image_count)  # pixel count of every image
    sum_pixel_count = 0  # record sum of pixels of images

    with tqdm(total=len(os.listdir(image_path)), unit=" img") as pbar:
        for i in range(image_count):
            rgb_img = cv2.imread(os.path.join(image_path, image_name_list[i]))
            b_mean_array[i] = np.mean(rgb_img[:, :, 0])
            b_std_array[i] = np.std(rgb_img[:, :, 0])
            g_mean_array[i] = np.mean(rgb_img[:, :, 1])
            g_std_array[i] = np.std(rgb_img[:, :, 1])
            r_mean_array[i] = np.mean(rgb_img[:, :, 2])
            r_std_array[i] = np.std(rgb_img[:, :, 2])
            single_pixel_count[i] = rgb_img.shape[0] * rgb_img.shape[1]
            sum_pixel_count = sum_pixel_count + rgb_img.shape[0] * rgb_img.shape[1]
            pbar.update()

    # B
    mean_tem = 0
    for i in range(len(os.listdir(image_path))):
        mean_tem += single_pixel_count[i] * b_mean_array[i]
    b_mean = mean_tem / sum_pixel_count
    std_tem = 0
    for i in range(len(os.listdir(image_path))):
        std_tem += single_pixel_count[i] * (b_std_array[i] ** 2 + (b_mean - b_mean_array[i]) ** 2)
    b_std = math.sqrt(std_tem / sum_pixel_count)
    print("B_mean: {}  B_std: {}".format(b_mean, b_std))

    # G
    mean_tem = 0
    for i in range(len(os.listdir(image_path))):
        mean_tem += single_pixel_count[i] * g_mean_array[i]
    g_mean = mean_tem / sum_pixel_count
    std_tem = 0
    for i in range(len(os.listdir(image_path))):
        std_tem += single_pixel_count[i] * (g_std_array[i]**2 + (g_mean - g_mean_array[i]) ** 2)
    g_std = math.sqrt(std_tem / sum_pixel_count)
    print("G_mean: {}  G_std: {}".format(g_mean, g_std))

    # R
    mean_tem = 0
    for i in range(len(os.listdir(image_path))):
        mean_tem += single_pixel_count[i] * r_mean_array[i]
    r_mean = mean_tem / sum_pixel_count
    std_tem = 0
    for i in range(len(os.listdir(image_path))):
        std_tem += single_pixel_count[i] * (r_std_array[i]**2 + (r_mean - r_mean_array[i]) ** 2)
    r_std = math.sqrt(std_tem / sum_pixel_count)
    print("R_mean: {}  R_std: {}".format(r_mean, r_std))


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
    # image_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/origin/image"
    # gt_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/origin/gt"
    # save_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/segmented"
    image_path = "/home/xueruoyao/Documents/PythonProgram/dataset/deepglobe/image"
    gt_path = "/home/xueruoyao/Documents/PythonProgram/dataset/deepglobe/gt"
    save_path = "/home/xueruoyao/Documents/PythonProgram/dataset/new"

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
    image_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/segmented/train/gt"
    # image_path = "/home/xueruoyao/Documents/PythonProgram/dataset/data_aug/gt"
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


def data_split_csv():
    """ split original data to train, valid, test due to given csv files """
    # csv files should be named after train.csv, valid.csv, test.csv
    csv_dir_path = "/data/xueruoyao/csv"
    origin_data_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/origin"
    save_path = "/data/xueruoyao/dataset/road_extraction/deepglobe/1024_segmented"

    print("split data according to csv")
    if not os.path.exists(save_path):
        for dir_name in ["train", "valid", "test"]:
            os.makedirs(os.path.join(save_path, dir_name, "image"))
            os.makedirs(os.path.join(save_path, dir_name, "gt"))
            if len(os.listdir(os.path.join(save_path, dir_name, "image"))) != 0 or \
               len(os.listdir(os.path.join(save_path, dir_name, "gt"))) != 0:
                raise RuntimeError(f"save path {os.path.join(save_path, dir_name)} is not empty")

    train_csv = pd.read_csv(os.path.join(csv_dir_path, "train.csv"))
    valid_csv = pd.read_csv(os.path.join(csv_dir_path, "valid.csv"))
    test_csv = pd.read_csv(os.path.join(csv_dir_path, "test.csv"))
    origin_image_name_list = os.listdir(os.path.join(origin_data_path, "image"))

    with tqdm(total=len(train_csv)+len(valid_csv)+len(test_csv), unit=" img") as pbar:
        for i in range(len(train_csv)):
            csv_image_name = os.path.basename(train_csv.iloc[i][0])
            for origin_image_name in origin_image_name_list:
                if origin_image_name.split(".")[0] == csv_image_name.split("_")[0]:
                    shutil.copy2(os.path.join(os.path.join(origin_data_path, "image", origin_image_name)),
                                 os.path.join(os.path.join(save_path, "train", "image", origin_image_name)))
                    shutil.copy2(os.path.join(os.path.join(origin_data_path, "gt", origin_image_name)),
                                 os.path.join(os.path.join(save_path, "train", "gt", origin_image_name)))
                    pbar.update()
                    break

        for i in range(len(valid_csv)):
            csv_image_name = os.path.basename(valid_csv.iloc[i][0])
            for origin_image_name in origin_image_name_list:
                if origin_image_name.split(".")[0] == csv_image_name.split("_")[0]:
                    shutil.copy2(os.path.join(os.path.join(origin_data_path, "image", origin_image_name)),
                                 os.path.join(os.path.join(save_path, "valid", "image", origin_image_name)))
                    shutil.copy2(os.path.join(os.path.join(origin_data_path, "gt", origin_image_name)),
                                 os.path.join(os.path.join(save_path, "valid", "gt", origin_image_name)))
                    pbar.update()
                    break

        for i in range(len(test_csv)):
            csv_image_name = os.path.basename(test_csv.iloc[i][0])
            for origin_image_name in origin_image_name_list:
                if origin_image_name.split(".")[0] == csv_image_name.split("_")[0]:
                    shutil.copy2(os.path.join(os.path.join(origin_data_path, "image", origin_image_name)),
                                 os.path.join(os.path.join(save_path, "test", "image", origin_image_name)))
                    shutil.copy2(os.path.join(os.path.join(origin_data_path, "gt", origin_image_name)),
                                 os.path.join(os.path.join(save_path, "test", "gt", origin_image_name)))
                    pbar.update()
                    break
    print(f"csv info: train: {len(train_csv)}  valid: {len(valid_csv)}  test: {len(test_csv)}")
    print("save info: train: {}  valid: {}  test: {}".format(
        len(os.listdir(os.path.join(save_path, "train", "image"))),
        len(os.listdir(os.path.join(save_path, "valid", "image"))),
        len(os.listdir(os.path.join(save_path, "test", "image")))
    ))


if __name__ == "__main__":
    # data_clean()
    # data_split()
    # statistic_image_size()
    # compute_rgb_mean_std()
    # label_statistics()
    # find_damaged_label()
    data_split_csv()
    pass