# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 24　14:44

import cv2
import skimage
from skimage.filters import gabor_kernel
from skimage import  measure

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt




class TowerDetecter:
    '''
    Set up image needed to be recognize.
    '''

    def __init__(self, img, debug_flag=False):
        self.debug_flag = debug_flag
        # self.src_img = img
        scale = 1.0
        if img.shape[0] > 1000:
            scale = int(img.shape[0] / 500.0)

        # self.src_img = img
        self.src_img = cv2.resize(img,
                                  (int(img.shape[0] / scale), int(img.shape[1] / scale)),
                                  interpolation=cv2.INTER_CUBIC)
        self.img_name_list = list()
        self.img_list = list()

        self.tAddImg('src_img', self.src_img)

    def tAddImg(self, str_name, img):
        # self.img_name_list.append(str_name)
        # self.img_list.append(img)
        cv2.namedWindow(str_name, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(str_name, img)

    '''
    Pre-process the image
    '''

    def lower_preprocess(self):
        self.std_img = np.std(self.src_img.copy().astype(dtype=np.float), axis=2)
        self.std_img = self.std_img / np.max(self.std_img)
        # self.img_list.append(self.dis_img)
        # self.img_name_list.append('dis_img')
        self.tAddImg('dis_img', self.std_img)

        # self.processed_img = self.src_img
        self.processed_img = cv2.cvtColor(self.src_img,
                                          cv2.COLOR_BGR2HSV)

        self.tAddImg('processed_img', self.processed_img)
        grey_color = np.array([
            [0, 0, 40],
            [100, 100, 90]
        ])
        self.color_img = cv2.inRange(self.processed_img, grey_color[0], grey_color[1])
        self.tAddImg('color_img', self.color_img)

        self.strength_img = np.mean(self.src_img, axis=2)
        self.strength_img = self.strength_img / np.max(self.strength_img)
        # self.tAddImg('strength_img',self.strength_img)

        knn = cv2.createBackgroundSubtractorKNN()
        self.back_img = knn.apply(self.src_img)
        # self.tAddImg('back_img',self.back_img)
        tmp = ((self.std_img.copy() / np.max(self.std_img) * 254 + 1).astype(dtype=np.uint8).copy())
        # tmp = cv2.adaptiveThreshold(tmp, 100,
        #                             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                             blockSize=121,
        #                             thresholdType=cv2.THRESH_BINARY,
        #                             C=0)
        # self.threshold_
        # self.threshold_img = cv2.threshold(self.)
        self.canny_img = cv2.Canny(tmp,
                                   150,
                                   200
                                   )
        # self.canny_img = cv2.Laplacian(tmp, cv2.CV_8U)
        # self.canny_img = cv2.Laplacian(self.dis_img/np.max(self.dis_img), cv2.CV_64FC1)
        self.tAddImg('canny', self.canny_img)
        # cv2.imshow('canny', self.canny_img)
        self.tAddImg('tmp', tmp)

        self.line_img = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)

        lines = cv2.HoughLinesP(tmp, 3.0, np.pi / 180,
                                20,
                                minLineLength=self.src_img.shape[0] / 6,
                                maxLineGap=5)
        # print(lines.shape)
        # for x1, y1, x2, y2 in lines[0]:
        for index in range(lines.shape[0]):
            x1 = lines[index, 0, 0]
            y1 = lines[index, 0, 1]
            x2 = lines[index, 0, 2]
            y2 = lines[index, 0, 3]
            cv2.line(self.line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.tAddImg('line_img', self.line_img)

        self.orb_img = self.src_img.copy()
        orb = cv2.ORB_create()
        kp = orb.detect(self.orb_img, None)
        kp, des = orb.compute(self.orb_img, kp)

        self.orb_img = cv2.drawKeypoints(self.orb_img, kp, None, color=(0, 222, 0), flags=0)
        self.tAddImg('orb img', self.orb_img)

        # print(tmp.min(), tmp.max(), tmp.std())

        # plt.figure()
        # plt.imshow(self.strength_img+self.dis_img)
        # plt.colorbar()
        # plt.show()

    def preprocess(self):
        self.hsv_img = cv2.cvtColor(self.src_img,
                                    cv2.COLOR_RGB2HSV)
        self.tAddImg('hsvimg', self.hsv_img)

        self.std_img = np.std(self.src_img.copy().astype(dtype=np.float),
                              axis=2)
        self.bi_grey_img = np.ones_like(self.std_img,dtype=np.uint8)
        self.bi_grey_img[np.where(self.std_img<1)] = 0
        self.bi_grey_img[np.where(self.std_img>7)] = 0
        self.bi_grey_img *= 255
        self.tAddImg('bi img',self.bi_grey_img)

        # print(self.hsv_img.shape)
        # self.h = self.hsv_img[:,:,0]
        # self.s = self.hsv_img[:,:,1]
        # self.v = self.hsv_img[:,:,2]

        # self.tAddImg('h',self.h)
        # self.tAddImg('s',self.s)
        # self.tAddImg('v',self.v)

        self.grey_img = cv2.inRange(self.hsv_img,(0.0,0.0,46),
                                    (180,43,220))
        self.tAddImg('grey',self.grey_img)

        self.both_img = np.zeros(self.bi_grey_img,dtype=float)

        self.both_img[np.where(self.bi_grey_img>100)&np.where(self.grey_img>100)] = 255

        self.tAddImg('both',self.both_img)











    def pltShow(self, index=0):
        plt.figure(index)
        for i in range(len(self.img_list)):
            # if not 'proc' in self.img_name_list[i] :
            #     continue
            plt.subplot(3, len(self.img_list) / 3 + 1, i + 1)
            plt.imshow(self.img_list[i] / np.mean(self.img_list[i]))
            # plt.colorbar()
            plt.title(self.img_name_list[i])
