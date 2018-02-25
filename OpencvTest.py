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
        ti+=1
        # if not ti is 6:
        #     continue
        print(name)
        t_name = 'image\\' + name
        im = cv2.imread(t_name)
        td = PowerTowerDetector.TowerDetecter(im, True)
        # td.preprocess()
        # td.multiLayerProcess()
        td.contour_process()
        # td.hsvProcess()

        cv2.waitKey()

        # cv2.imwrite('res_image\\'+name, td.v_line_img)

        detector_list.append(td)
        # break
    plt.show()