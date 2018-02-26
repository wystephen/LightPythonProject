# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 26　11:09
import cv2
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import os

import PowerTowerDetector

from sklearn import svm
from sklearn.metrics import *

if __name__ == '__main__':

    detector_list = list()
    ti = 0
    total_feature_list = list()
    total_label_list = list()
    num_counter = 0

    # load all image in the dataset
    for name in os.listdir('./image'):
        ti += 1
        print(name)
        t_name = 'image\\' + name
        im = cv2.imread(t_name)
        label_img = cv2.imread('label_image\\Label_'
                               + str(ti) + '.png')
        td = PowerTowerDetector.TowerDetecter(im, False)

        feature_list, label_list = td.dataset_builder(win_size_list=[400],
                                                      label_img=label_img.copy())
        num_counter += len(feature_list)
        print(len(total_feature_list))
        total_feature_list.extend(feature_list)
        total_label_list.extend(label_list)

        cv2.waitKey()
        detector_list.append(td)

    print('num counter ', num_counter)
    print(len(total_feature_list), len(total_label_list))
    data_x = np.zeros([len(total_feature_list), total_feature_list[0].shape[0]])
    data_y = np.zeros([len(total_label_list), 1])
    for i in range(len(total_feature_list)):
        data_y[i, 0] = total_label_list[i]
        data_x[i, :] = total_feature_list[i][:]
    print(data_x.shape, data_y.shape)

    np.savetxt('data_x.csv', data_x, delimiter=',')
    np.savetxt('data_y.csv', data_y, delimiter=',')

    print(np.max(data_x[:, :60]), np.min(data_x[:, :60]))

    # clf = svm.SVC(kernel='rbf')
    # clf.fit(data_x, data_y.copy())
    # pre_y = clf.predict(data_x)
    # print(accuracy_score(pre_y,data_y.copy()))
    # print('error numbers:', np.sum(np.abs(data_y - pre_y)))
    #
    # plt.figure()
    # plt.plot(data_y, label='data y')
    # plt.plot(pre_y,label='prey')
    # plt.legend()

    plt.show()
