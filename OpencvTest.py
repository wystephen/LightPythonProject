# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 24　12:59

import cv2
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import os

import PowerTowerDetector

if __name__ == '__main__':

    detector_list = list()
    ti = 0
    # load all image in the dataset
    for name in os.listdir('./image'):
        print(name)
        t_name = 'image\\' + name
        im = cv2.imread(t_name)
        td = PowerTowerDetector.TowerDetecter(im, False)
        td.preprocess()

        # cv2.waitKey()
        ti+=1
        td.pltShow(ti)

        detector_list.append(td)
        break
    plt.show()