# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 26　11:09
import cv2
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import os

import PowerTowerDetector

if __name__ == '__main__':

    detector_list = list()
    ti = 0
    total_feature_list = list()
    total_label_list = list()

    # load all image in the dataset
    for name in os.listdir('./image'):
        ti += 1
        # if not ti is 6:
        #     continue
        print(name)
        t_name = 'image\\' + name
        im = cv2.imread(t_name)
        label_img = cv2.imread('label_image\\'
                           +name.split('.')[0]+'.png')
        td = PowerTowerDetector.TowerDetecter(im, True)

        feature_list, label_list = td.dataset_builder(label_img=label_img)
        total_feature_list.extend(feature_list)
        total_label_list.extend(label_list)


        # td.preprocess()
        # td.multiLayerProcess()
        # td.contour_process()
        # td.keyPointProcess()
        # td.hsvProcess()

        cv2.waitKey()

        # cv2.imwrite('res_image\\'+name, td.v_line_img)

        detector_list.append(td)
        # break
    print(len(feature_list),len(label_list))
    data_x = np.zeros([len(feature_list),feature_list[0].shape[0]])
    data_y = np.zeros([len(label_list),1])
    for i in range(len(feature_list)):
        data_y[i,0] = label_list[i]
        data_x[i,:] = feature_list[i][:]

    np.savetxt('data_x.csv',data_x,delimiter=',')
    np.savetxt('data_y.csv',data_y,delimiter=',')

    plt.show()