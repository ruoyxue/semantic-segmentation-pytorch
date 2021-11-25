import cv2
import os
import math
import numpy as np
import time


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


def compute_rgb_mean_std(image_path: str):
    """ calculate mean and std of R, G, B channel"""
    num = 0  # image sum
    R_mean, R_std = [], []
    G_mean, G_std = [], []
    B_mean, B_std = [], []
    size = []  # pixels of every image
    sum_size = 0  # 记录所有图片总像素
    start = time.time()
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
        num = num + 1
        if num % 50 == 0:
            print("num = {}".format(num))

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

    end = time.time()
    print("total time is ", format(end - start, '.2f'), " s\n")
