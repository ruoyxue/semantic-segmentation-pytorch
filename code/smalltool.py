import cv2
import os
import numpy as np


def label_category_find():
    # calculate all the categories of label
    import cv2
    labels = []
    count = 0
    for label_name in os.listdir("../dataset/semantic_segmentation/label"):
        tem = (cv2.imread(os.path.join("../dataset/semantic_segmentation/label", label_name)))
        labels += list(np.unique(tem[:, :, 2]))
        count += 1
        print(count)

    print(np.unique(np.array(labels)))
