# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 26　15:25
import cv2
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import os

import PowerTowerDetector

from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import *

if __name__ == '__main__':

    detector_list = list()
    ti = 0
    clf_model = joblib.load('model/svm/train_model.m')

    for name in os.listdir('./image'):
        ti += 1
        # if not ti is 6:
        #     continue
        print(name)
        t_name = 'image\\' + name
        im = cv2.imread(t_name)
        td = PowerTowerDetector.TowerDetecter(im, True)

        label_img = cv2.imread('label_image\\Label_'
                               + str(ti) + '.png')

        td.sub_regeion_classify(clf_model=clf_model,
                                label_img=label_img,
                                win_size_list=[500])

        cv2.imwrite('./res_img/' + name, td.original_img)

        cv2.waitKey()
