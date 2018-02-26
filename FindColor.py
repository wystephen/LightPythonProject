# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 24　23:03
import cv2
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import os

import PowerTowerDetector

if __name__ == '__main__':

    detector_list = list()
    ti = 0

    image_list = list()
    # load all image in the dataset
    for name in os.listdir('./image'):
        ti+=1
        if not ti is 5:
            continue
        print(name)
        t_name = 'image\\' + name
        im = cv2.imread(t_name)
        image_list.append(im)

    print('after load image')

    plt.figure(1)
    for i in range(len(image_list)):
        img = image_list[i]
        plt.subplot(2,3,i+1)
        plt.imshow(img)
    # cv2.createLineSegmentDetector()


    plt.show()