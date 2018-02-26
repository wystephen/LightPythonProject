# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 24　12:59

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
    # clf_model = joblib.load('svm.m')
    # load all image in the dataset
    total_feature = list()

    for name in os.listdir('./image'):
        ti+=1
        # if not ti is 6:
        #     continue
        print(name)
        t_name = 'image\\' + name
        im = cv2.imread(t_name)
        td = PowerTowerDetector.TowerDetecter(im, False)

        label_img = cv2.imread('label_image\\Label_'
                               + str(ti) + '.png')

        td.sub_regeion_classify(clf_model=clf_model,
                                label_img=label_img,
                                win_size_list=[500])

        total_feature.extend(td.feature_list)

        cv2.waitKey()

        # cv2.imwrite('res_image\\'+name, td.v_line_img)

        detector_list.append(td)
        # break
    data_feature = np.zeros([len(total_feature),total_feature[0].shape[0]])
    for i in range(data_feature.shape[0]):
        data_feature[i,:] = total_feature[i][:]
    data_x = np.loadtxt('data_x.csv',delimiter=',')
    data_y = np.loadtxt('data_y.csv',delimiter=',')
    pre_y_feature = clf_model.predict(data_feature)
    pre_y_data = clf_model.predict(data_x)

    print('diff of feature and data:', np.sum(np.abs(pre_y_feature-pre_y_data)), pre_y_feature.shape)
    print('feature:', np.sum(np.abs(pre_y_feature-data_y)), pre_y_feature.shape)
    print('data:',np.sum(np.abs(data_y-pre_y_data)), pre_y_feature.shape)
    print('feature:',accuracy_score(data_y,pre_y_feature))
    print('data:', accuracy_score(data_y,pre_y_data))

    # the_clf = svm.SVC(kernel='rbf')
    # the_clf.fit(data_feature, data_y)
    # print(accuracy_score(data_y,the_clf.predict(data_feature)))
    # joblib.dump(the_clf,'svm.m')


    # plt.figure()
    # plt.imshow(data_x-data_feature)
    # plt.colorbar()
    # plt.show()
    # plt.figure()
    # plt.plot(pre_y_data,label='preydata')
    # plt.plot(pre_y_feature,label='preyfeature')
    #
    # plt.grid()
    # plt.legend()
    #
    # plt.show()
