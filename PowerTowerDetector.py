# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 24　14:44

import cv2
import skimage
from skimage.filters import gabor_kernel
from skimage import measure
from skimage import transform

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

import math

from HogDescriptor import Hog_descriptor

from sklearn import svm
from sklearn.externals import joblib


class TowerDetecter:
    def __init__(self, img, debug_flag=False):
        '''

        :param img:  input image
        :param debug_flag: if true, the image will display through cv.imshow()
        '''
        self.debug_flag = debug_flag
        self.original_img = img.copy()
        # self.src_img = img
        scale = 1.0
        scale_to_width = 1000
        if img.shape[0] > scale_to_width:
            scale = int(img.shape[0] / scale_to_width)

        # self.src_img = img
        self.src_img = cv2.resize(img,
                                  (int(img.shape[1] / scale), int(img.shape[0] / scale)),
                                  interpolation=cv2.INTER_CUBIC)
        self.img_name_list = list()
        self.img_list = list()

        self.tAddImg('src_img', self.src_img)

    def tAddImg(self, str_name, img):
        '''
        display image with modified windows size.
        :param str_name: name of the image,
        :param img: image for display
        :return:
        '''
        if self.debug_flag:
            cv2.namedWindow(str_name, cv2.WINDOW_GUI_NORMAL)
            cv2.imshow(str_name, img)

    def sub_regeion_classify(self, clf_model,
                             win_size_list=[200],
                             label_img=None):
        self.feature_list = list()
        tmp_src_img = self.original_img.copy()
        for win_size in win_size_list:
            tmp_res_img = self.original_img.copy()
            tmp_mask_img = np.zeros(
                [int(self.original_img.shape[0] / win_size + 1), int(self.original_img.shape[1] / win_size + 1)],
                dtype=np.uint8)
            for i in range(0, self.original_img.shape[0], win_size):
                for j in range(0, self.original_img.shape[1], win_size):
                    end_x = min(i + win_size, self.original_img.shape[0])
                    end_y = min(j + win_size, self.original_img.shape[1])
                    feature = self.feature_extract(tmp_src_img[i:end_x, j:end_y])
                    self.feature_list.append(feature)
                    # print(feature)
                    y = clf_model.predict(feature.reshape([1, -1]))
                    if y[0] == 1:
                        self.original_img = cv2.rectangle(self.original_img,
                                                          (j, i),
                                                          (end_y, end_x),
                                                          (0, 0, 255),
                                                          thickness=10)

                    # if label_img is None:
                    #     dalfjdsalkfjlkasdlkfjoi=1
                    # else:
                    #     if float(np.count_nonzero(label_img[i:end_x, j:end_y])) > 0.7 * (
                    #             float(end_x - i) * float(end_y - j)):
                    #         self.original_img = cv2.rectangle(self.original_img,
                    #                                           (j, i),
                    #                                           (end_y, end_x),
                    #                                           (0, 255, 0),
                    #                                           thickness=5)

        self.tAddImg('marked', self.original_img)

    def multiLayerProcess(self, win_size_list=[200]):
        self.result_imgs = list()

        for win_size in win_size_list:
            tmp_res_img = self.original_img.copy()
            tmp_mask_img = np.zeros(
                [int(self.original_img.shape[0] / win_size + 1), int(self.original_img.shape[1] / win_size + 1)],
                dtype=np.uint8)
            for i in range(0, self.original_img.shape[0], win_size):
                for j in range(0, self.original_img.shape[1], win_size):
                    end_x = min(i + win_size, self.original_img.shape[0])
                    end_y = min(j + win_size, self.original_img.shape[1])
                    tmp_res_img[i:end_x, j:end_y, :], is_tower_flag = \
                        self.sub_image_process(self.original_img[i:end_x, j:end_y, :])
                    if is_tower_flag:
                        tmp_mask_img[int(i / win_size), int(j / win_size)] = 255
                ## process tmp mask img
            # tmp_mask_img = cv2.erode(tmp_mask_img,cv2.getStructuringElement(
            #     cv2.MORPH_RECT,
            #     (3,3)
            # ))
            tmp_mask_img = cv2.morphologyEx(
                tmp_mask_img,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (2, 2)
                ))

            for i in range(0, self.original_img.shape[0], win_size):
                for j in range(0, self.original_img.shape[1], win_size):
                    end_x = min(i + win_size, self.original_img.shape[0])
                    end_y = min(j + win_size, self.original_img.shape[1])
                    if tmp_mask_img[int(i / win_size), int(j / win_size)] > 2:
                        # print('mask')
                        tmp_res_img[i:end_x, j:end_y, :] = cv2.rectangle(
                            self.original_img[i:end_x, j:end_y, :].copy(), (0, 0),
                            (end_x - i - 1, end_y - j - 1), (0, 0, 255), 5).copy()

            self.tAddImg(str(win_size) + 'tmp_res', tmp_res_img)
            self.result_imgs.append(tmp_res_img)

    def sub_image_process(self, img):
        '''
        process sub img
        :param img:
        :return:
        '''
        height = img.shape[0]
        width = img.shape[1]

        self.sub_img = cv2.resize(img, (100, 100))
        b = self.sub_img[:, :, 0]
        g = self.sub_img[:, :, 1]
        r = self.sub_img[:, :, 2]

        self.sub_hsv_img = cv2.cvtColor(self.sub_img, cv2.COLOR_RGB2HSV)
        h = self.sub_hsv_img[:, :, 0]
        s = self.sub_hsv_img[:, :, 1]
        v = self.sub_hsv_img[:, :, 2]

        self.h_canny = cv2.Canny(h, 100, 200)
        self.s_canny = cv2.Canny(s, 100, 200)
        self.v_canny = cv2.Canny(v, 100, 200)

        self.s_canny *= 0

        h_count = np.count_nonzero(self.h_canny)
        v_count = np.count_nonzero(self.v_canny)
        # print(h_count, v_count)
        # color_mask = h

        # if h_count + v_count > 50 \
        #         and h_count < 0.7 * (h_count + v_count) \
        #         and v_count < 0.7 * (h_count + v_count):
        #     color_mask = np.zeros_like(color_mask)
        # else:
        #     color_mask = np.ones_like(color_mask) * 255

        self.h_line = np.zeros_like(self.h_canny)
        self.s_line = np.zeros_like(self.s_canny)
        self.v_line = np.zeros_like(self.v_canny)
        self.h_line = self.detectAnddraw(self.h_canny)
        self.s_line = self.detectAnddraw(self.s_canny)
        self.v_line = self.detectAnddraw(self.v_canny)

        self.sub_std_img = np.std(self.sub_img, axis=2)
        self.grey_mask_img = np.ones_like(self.sub_std_img, dtype=np.uint8)
        # grey must exist.
        self.grey_mask_img[np.where(self.sub_std_img < 1.0)] = 0
        self.grey_mask_img[np.where(self.sub_std_img > 7.0)] = 0

        self.red_mask_img = np.zeros_like(self.grey_mask_img)
        self.green_mask_img = np.zeros_like(self.grey_mask_img)

        for i in range(self.sub_img.shape[0]):
            for j in range(self.sub_img.shape[1]):
                if r[i, j] > 200 and b[i, j] + g[i, j] < 100:
                    self.red_mask_img[i, j] = 255
                if g[i, j] > 200 and b[i, j] + g[i, j] < 100:
                    self.green_mask_img[i, j] = 255

        # self.sub_img[:,:,:1] =  0
        is_tower_flag = False
        if (np.count_nonzero(self.h_line) > 40 or \
            np.count_nonzero(self.s_line) > 40 or \
            np.count_nonzero(self.v_line) > 40) and \
                np.count_nonzero(self.grey_mask_img) > 10 and \
                np.count_nonzero(self.red_mask_img) < 3000 and \
                np.count_nonzero(self.green_mask_img) < 9900:
            # cv2.rectangle(self.sub_img, (0, 0), (height, width), (0, 0, 223), 5)
            is_tower_flag = True

        return cv2.resize(self.sub_img, (width, height)), is_tower_flag

    def detectAnddraw(self, img,
                      minLineLength=50,
                      maxLineGap=5,
                      threshold=30,
                      other_condition=True):

        # 直线检测，（电线杆上有直线，自然界直线比较少）
        line_list = (transform.probabilistic_hough_line(img, threshold=threshold,
                                                        line_length=minLineLength,
                                                        line_gap=maxLineGap))
        lines = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        if other_condition:
            angle_array = np.zeros(int(180 / 5) + 1)
            for (x1, y1), (x2, y2) in line_list:
                a = np.arctan2(y2 - y1, x2 - x1)
                a = int(abs(a / np.pi * 180.0))
                angle_array[int(a / 5)] += 1
            counter = 0
            for angle in angle_array:
                if angle > 0:
                    counter += 1

            if counter < 4 and len(line_list) < 15:
                for (x1, y1), (x2, y2) in line_list:
                    cv2.line(lines, (x1, y1), (x2, y2), (255, 255, 255), 2)
        else:
            for (x1, y1), (x2, y2) in line_list:
                cv2.line(lines, (x1, y1), (x2, y2), (255, 255, 255), 2)

        return cv2.cvtColor(lines, cv2.COLOR_RGB2GRAY)

    def detectTowerLine(self, src_img,
                        img,
                        minLineLength=50,
                        maxLineGap=5,
                        threshold=30,
                        other_condition=True):

        # 直线检测，（电线杆上有直线，自然界直线比较少）
        line_list = (transform.probabilistic_hough_line(img, threshold=threshold,
                                                        line_length=minLineLength,
                                                        line_gap=maxLineGap))
        lines = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for (x1, y1), (x2, y2) in line_list:
            cv2.line(lines, (x1, y1), (x2, y2), (255, 255, 255), 2)

        return cv2.cvtColor(lines, cv2.COLOR_RGB2GRAY), line_list

    '''
    process the image
    '''

    def preprocess(self):
        # 用hSV 色彩空间表示 图像
        self.hsv_img = cv2.cvtColor(self.src_img,
                                    cv2.COLOR_RGB2HSV)
        self.tAddImg('hsvimg', self.hsv_img)
        # hsv各个分量
        self.h = self.hsv_img[:, :, 0]
        self.s = self.hsv_img[:, :, 1]
        self.v = self.hsv_img[:, :, 2]

        # rgb 各个分量
        self.r = self.src_img[:, :, 0]
        self.g = self.src_img[:, :, 1]
        self.b = self.src_img[:, :, 2]

        # 每个像素三个通道的值的标准差（灰色r=g=b），所以越小越接近灰色
        self.std_img = np.std(self.src_img.copy().astype(dtype=np.float),
                              axis=2)

        self.bi_grey_img = np.ones_like(self.std_img, dtype=np.uint8)
        self.bi_grey_img[np.where(self.std_img < 1)] = 0  # 排除白色（0，0，0）
        self.bi_grey_img[np.where(self.std_img > 7)] = 0  # 排除非灰色系
        self.bi_grey_img *= 255
        self.tAddImg('bi img', self.bi_grey_img)

        self.grey_img = cv2.inRange(self.hsv_img, (0.0, 0.0, 76),
                                    (180, 33, 240))
        self.tAddImg('grey', self.grey_img)

        self.both_img = np.zeros_like(self.bi_grey_img, dtype=np.uint8)  # 根据颜色判断可能是电线塔的像素
        self.wrong_color_img = np.zeros_like(self.bi_grey_img, dtype=np.uint8)  # 根据颜色判断不可能是电线塔的像素

        for i in range(self.both_img.shape[0]):
            for j in range(self.both_img.shape[1]):
                if self.bi_grey_img[i, j] > 100 and self.grey_img[i, j] > 100 and \
                        (self.r[i, j] + self.g[i, j]) < 0.9 * (self.r[i, j] + self.g[i, j] + self.b[i, j]) and \
                        self.g[i, j] < 0.95 * (self.r[i, j] + self.g[i, j] + self.b[i, j]):
                    # 条件1： 在hsv色彩空间和 rgb色彩空间都表现为灰色
                    # 条件2： 红色和绿色（混合为黄色）之和小于0。9。（排除土地）
                    # 条件3： 绿色 占比小于 0。95。（排除绿色植物）
                    self.both_img[i, j] = 255
                elif self.g[i, j] > 0.99 * (self.r[i, j] + self.g[i, j] + self.b[i, j]):
                    # 纯绿色不可能是电线塔
                    self.wrong_color_img[i, j] = 255
                elif self.r[i, j] + self.g[i, j] > 0.95 * (self.r[i, j] + self.g[i, j] + self.b[i, j]) and \
                        abs(self.r[i, j] - self.g[i, j]) < 0.05 * ((self.r[i, j] + self.g[i, j] + self.b[i, j])):
                    # 黄色，不可能是电线塔
                    self.wrong_color_img[i, j] = 255
                elif self.r[i, j] > 0.9 * ((self.r[i, j] + self.g[i, j] + self.b[i, j])):
                    # 红色， 不可能是电线塔。
                    self.wrong_color_img[i, j] = 255

        self.tAddImg('both', self.both_img)

        # 二值化并形态学闭运算。
        ret, self.threshold_img = cv2.threshold(self.both_img, 100, 255, cv2.THRESH_BINARY)
        self.tAddImg('threshold', self.threshold_img)

        self.morph_img = cv2.morphologyEx(self.threshold_img, cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

        self.tAddImg('closed', self.morph_img)

    def hsvProcess(self):
        self.hsv_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2HSV)
        b = self.src_img[:, :, 0]
        g = self.src_img[:, :, 1]
        r = self.src_img[:, :, 2]

        h = self.hsv_img[:, :, 0]
        s = self.hsv_img[:, :, 1]
        v = self.hsv_img[:, :, 2]

        self.h_canny = cv2.Canny(h, 100, 200)
        self.s_canny = cv2.Canny(s, 100, 200)
        self.v_canny = cv2.Canny(v, 100, 200)
        # self.h_line_img = self.detectTowerLine(self.src_img, self.h_canny, 100, 10, 30, False)
        # self.s_line_img = self.detectTowerLine(self.src_img, self.s_canny, 100, 10, 30, False)
        self.v_line_img, v_line_list = self.detectTowerLine(self.src_img,
                                                            self.v_canny,
                                                            150, 10, 30, False)

        # self.v_line_img  = cv2.morphologyEx(
        #     self.v_line_img,
        #     cv2.MORPH_OPEN,
        #     cv2.getStructuringElement(
        #         cv2.MORPH_CROSS,
        #         (5,5)
        #     )
        # )
        # for i in range(1, self.v_line_img.shape[0] - 1):
        #     for j in range(1, self.v_line_img.shape[1] - 1):
        #         if g[i, j] > r[i, j] + b[i, j] and \
        #                 np.sum(g[i - 1:i + 1, j - 1:j + 1]) > np.sum(r[i - 1:i + 1, j - 1:j + 1] +
        #                                                              b[i - 1:i + 1, j - 1:j + 1]):
        #             self.v_line_img[i, j] = 0
        ret, binary = cv2.threshold(self.v_line_img, 100, 255, cv2.THRESH_BINARY)
        im, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
        self.contour_img = cv2.drawContours(self.src_img.copy(),
                                            contours,
                                            -1,
                                            (0, 0, 255), 3)
        for cnt in contours:
            if len(cnt) > 5:
                approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)
                if len(approx) == 3:
                    cv2.drawContours(self.contour_img, [cnt], 0, (0, 255, 0), 2)

                # cv2.ellipse(self.contour_img,ellipsis,(0,255,0),2)

        # self.tAddImg('h line', self.h_line_img)
        # self.tAddImg('s line', self.s_line_img)
        self.tAddImg('vline ', self.v_line_img)
        self.tAddImg('contour', self.contour_img)

    '''
    process the image
    '''

    def preprocess2(self):
        # 用hSV 色彩空间表示 图像
        self.hsv_img = cv2.cvtColor(self.src_img,
                                    cv2.COLOR_RGB2HSV)
        self.tAddImg('hsvimg', self.hsv_img)
        # hsv各个分量
        self.h = self.hsv_img[:, :, 0]
        self.s = self.hsv_img[:, :, 1]
        self.v = self.hsv_img[:, :, 2]

        # rgb 各个分量
        self.r = self.src_img[:, :, 0]
        self.g = self.src_img[:, :, 1]
        self.b = self.src_img[:, :, 2]

        # 每个像素三个通道的值的标准差（灰色r=g=b），所以越小越接近灰色
        self.std_img = np.std(self.src_img.copy().astype(dtype=np.float),
                              axis=2)

        self.bi_grey_img = np.ones_like(self.std_img, dtype=np.uint8)
        self.bi_grey_img[np.where(self.std_img < 1)] = 0  # 排除白色（0，0，0）
        self.bi_grey_img[np.where(self.std_img > 7)] = 0  # 排除非灰色系
        self.bi_grey_img *= 255
        self.tAddImg('bi img', self.bi_grey_img)

        self.grey_img = cv2.inRange(self.hsv_img, (0.0, 0.0, 76),
                                    (180, 33, 240))
        self.tAddImg('grey', self.grey_img)

        self.both_img = np.zeros_like(self.bi_grey_img, dtype=np.uint8)  # 根据颜色判断可能是电线塔的像素
        self.wrong_color_img = np.zeros_like(self.bi_grey_img, dtype=np.uint8)  # 根据颜色判断不可能是电线塔的像素

        for i in range(self.both_img.shape[0]):
            for j in range(self.both_img.shape[1]):
                if self.bi_grey_img[i, j] > 100 and self.grey_img[i, j] > 100 and \
                        (self.r[i, j] + self.g[i, j]) < 0.9 * (self.r[i, j] + self.g[i, j] + self.b[i, j]) and \
                        self.g[i, j] < 0.95 * (self.r[i, j] + self.g[i, j] + self.b[i, j]):
                    # 条件1： 在hsv色彩空间和 rgb色彩空间都表现为灰色
                    # 条件2： 红色和绿色（混合为黄色）之和小于0。9。（排除土地）
                    # 条件3： 绿色 占比小于 0。95。（排除绿色植物）
                    self.both_img[i, j] = 255
                elif self.g[i, j] > 0.99 * (self.r[i, j] + self.g[i, j] + self.b[i, j]):
                    # 纯绿色不可能是电线塔
                    self.wrong_color_img[i, j] = 255
                elif self.r[i, j] + self.g[i, j] > 0.95 * (self.r[i, j] + self.g[i, j] + self.b[i, j]) and \
                        abs(self.r[i, j] - self.g[i, j]) < 0.05 * ((self.r[i, j] + self.g[i, j] + self.b[i, j])):
                    # 黄色，不可能是电线塔
                    self.wrong_color_img[i, j] = 255
                elif self.r[i, j] > 0.9 * ((self.r[i, j] + self.g[i, j] + self.b[i, j])):
                    # 红色， 不可能是电线塔。
                    self.wrong_color_img[i, j] = 255

        self.tAddImg('both', self.both_img)

        # 二值化并形态学闭运算。
        ret, self.threshold_img = cv2.threshold(self.both_img, 100, 255, cv2.THRESH_BINARY)
        self.tAddImg('threshold', self.threshold_img)

        self.morph_img = cv2.morphologyEx(self.threshold_img, cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        self.tAddImg('closed', self.morph_img)

        result = self.src_img.copy()
        # 经验参数
        minLineLength = 130
        maxLineGap = 30
        # 直线检测，（电线杆上有直线，自然界直线比较少）
        lines = (transform.probabilistic_hough_line(self.morph_img, threshold=50,
                                                    line_length=minLineLength,
                                                    line_gap=maxLineGap))
        self.line_mask_img = np.zeros_like(self.src_img)
        for (x1, y1), (x2, y2) in lines:
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(self.line_mask_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

        self.tAddImg('lines', result)

        ret, self.threshold_line_img = cv2.threshold(cv2.cvtColor(self.line_mask_img, cv2.COLOR_RGB2GRAY),
                                                     100, 255, cv2.THRESH_BINARY)

        self.tAddImg('threshold line img', self.threshold_line_img)

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask = np.zeros_like(self.threshold_line_img, np.uint8)
        mask[np.where(self.threshold_line_img > 0)] = cv2.GC_FGD
        mask[np.where(self.wrong_color_img > 200)] = cv2.GC_BGD
        # 根据已经获得的一定是电线杆（前景fgd）和一定是背景（bgd）的部分通过grabcut算法获取电线杆轮廓
        mask, bgdModel, fgdModel = cv2.grabCut(self.src_img, mask,
                                               None, bgdModel,
                                               fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        mask = self.src_img * mask[:, :, np.newaxis]
        self.result_img = self.src_img.copy()  # .cvtColor(self.src_img, cv2.COLOR_RGB2RGBA)
        self.tAddImg('mask', mask)
        self.tAddImg('wrong img', self.wrong_color_img)

        windows_size = 50
        step_len = 10

        self.tmp_mask_layer_img = np.zeros_like(self.src_img)

        # 根据区域内（大小由windows——size确定）前景的像素的数目确定是否标记为电线塔。
        fil = np.ones([windows_size, windows_size])
        print('mask shape ', mask.shape)
        self.total_line_point = cv2.filter2D(mask[:, :, 0], -1, fil)
        print(self.total_line_point.shape)
        median_num = np.mean(self.total_line_point)
        tmp_red_img = np.zeros_like(self.tmp_mask_layer_img[:, :, 2])
        tmp_red_img[np.where(self.total_line_point > median_num)] = 255
        self.tmp_mask_layer_img[:, :, 2] = tmp_red_img

        # self.smooth_img = cv2.erode(self.smooth_img,cv2.getStructuringElement(cv2.MORPH_RECT,(30,30)))
        # self.tAddImg('smoothed', self.smooth_img)
        # 混合标记和原始图像（使得被识别为电线塔的区域蒙上红色mask）
        self.result_img = cv2.addWeighted(self.result_img, 0.6, self.tmp_mask_layer_img, 0.4, 0)
        self.tAddImg('result', self.result_img)

    def contour_process(self):
        # 用hSV 色彩空间表示 图像
        self.hsv_img = cv2.cvtColor(self.src_img,
                                    cv2.COLOR_RGB2HSV)
        self.tAddImg('hsvimg', self.hsv_img)
        # hsv各个分量
        self.h = self.hsv_img[:, :, 0]
        self.s = self.hsv_img[:, :, 1]
        self.v = self.hsv_img[:, :, 2]

        # rgb 各个分量
        self.r = self.src_img[:, :, 0]
        self.g = self.src_img[:, :, 1]
        self.b = self.src_img[:, :, 2]

        # 每个像素三个通道的值的标准差（灰色r=g=b），所以越小越接近灰色
        self.std_img = np.std(self.src_img.copy().astype(dtype=np.float),
                              axis=2)

        self.bi_grey_img = np.ones_like(self.std_img, dtype=np.uint8)
        self.bi_grey_img[np.where(self.std_img < 1)] = 0  # 排除白色（0，0，0）
        self.bi_grey_img[np.where(self.std_img > 7)] = 0  # 排除非灰色系
        self.bi_grey_img *= 255
        self.tAddImg('bi img', self.bi_grey_img)

        self.grey_img = cv2.inRange(self.hsv_img, (0.0, 0.0, 76),
                                    (180, 33, 240))
        self.tAddImg('grey', self.grey_img)

        self.both_img = np.zeros_like(self.bi_grey_img, dtype=np.uint8)  # 根据颜色判断可能是电线塔的像素
        self.wrong_color_img = np.zeros_like(self.bi_grey_img, dtype=np.uint8)  # 根据颜色判断不可能是电线塔的像素

        for i in range(self.both_img.shape[0]):
            for j in range(self.both_img.shape[1]):
                if self.bi_grey_img[i, j] > 100 and self.grey_img[i, j] > 100 and \
                        (self.r[i, j] + self.g[i, j]) < 0.9 * (self.r[i, j] + self.g[i, j] + self.b[i, j]) and \
                        self.g[i, j] < 0.95 * (self.r[i, j] + self.g[i, j] + self.b[i, j]):
                    # 条件1： 在hsv色彩空间和 rgb色彩空间都表现为灰色
                    # 条件2： 红色和绿色（混合为黄色）之和小于0。9。（排除土地）
                    # 条件3： 绿色 占比小于 0。95。（排除绿色植物）
                    self.both_img[i, j] = 255
                elif self.g[i, j] > 0.99 * (self.r[i, j] + self.g[i, j] + self.b[i, j]):
                    # 纯绿色不可能是电线塔
                    self.wrong_color_img[i, j] = 255
                elif self.r[i, j] + self.g[i, j] > 0.95 * (self.r[i, j] + self.g[i, j] + self.b[i, j]) and \
                        abs(self.r[i, j] - self.g[i, j]) < 0.05 * ((self.r[i, j] + self.g[i, j] + self.b[i, j])):
                    # 黄色，不可能是电线塔
                    self.wrong_color_img[i, j] = 255
                elif self.r[i, j] > 0.9 * ((self.r[i, j] + self.g[i, j] + self.b[i, j])):
                    # 红色， 不可能是电线塔。
                    self.wrong_color_img[i, j] = 255

        # self.both_img = cv2.dilate(
        #     self.both_img,
        #     cv2.
        # )
        a, contour, he = cv2.findContours(self.both_img,
                                          cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
        self.contour_img = self.src_img
        for cnt in contour:
            approx = cv2.approxPolyDP(cnt,
                                      0.1 * cv2.arcLength(cnt, True),
                                      True)
            if len(approx) == 3:
                cv2.drawContours(self.contour_img,
                                 cnt, 0, (0, 0, 255), 3)

        self.tAddImg('both', self.both_img)
        self.tAddImg('contour', self.contour_img)

    def dataset_builder(self,
                        win_size_list=[200],
                        label_img=None):
        '''

        :param win_size_list:
        :param label_img:
        :return:
        '''
        feature_list = list()
        label_list = list()
        tmp_src_img = self.original_img.copy()

        for win_size in win_size_list:
            tmp_mask_img = np.zeros(
                [int(self.original_img.shape[0] / win_size + 1), int(self.original_img.shape[1] / win_size + 1)],
                dtype=np.uint8)
            for i in range(0, self.original_img.shape[0], win_size):
                for j in range(0, self.original_img.shape[1], win_size):
                    end_x = min(i + win_size, self.original_img.shape[0])
                    end_y = min(j + win_size, self.original_img.shape[1])
                    feature = self.feature_extract(
                        tmp_src_img[i:end_x,j:end_y]
                    )
                    #     all_f[k, :] = self.feature_extract(
                    #         self.original_img[i:end_x, j:end_y]
                    #     )
                    # from scipy.spatial.distance import pdist
                    # max_dis = np.max(pdist(all_f))
                    # if max_dis > 0.1:
                    #     print('max dis:', max_dis)

                    feature_list.append(feature)
                    if label_img is None:
                        label_list.append(0)
                    else:
                        # counter = np.where(label_img[i:end_x, j:end_y] > 0)
                        if float(np.count_nonzero(label_img[i:end_x, j:end_y])) > 0.7 * (
                                float(end_x - i) * float(end_y - j)):
                            label_list.append(1.0)
                            self.original_img = cv2.rectangle(self.original_img,
                                                              (j, i),
                                                              (end_y, end_x),
                                                              (0, 255, 0),
                                                              thickness=10)
                        else:
                            label_list.append(0.0)
        self.tAddImg('marked', self.original_img)
        return feature_list, label_list

    def hog(self, img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)

        bin_n = 5
        # quantizing binvalues in (0...16)
        bins = np.int32(bin_n * ang / (2 * np.pi))

        # Divide to 4 sub-squares
        bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        return hist

    def feature_extract(self, img):
        local_img = cv2.resize(img.copy(), (100, 100))
        local_img = cv2.cvtColor(local_img, cv2.COLOR_BGR2HSV)
        h = local_img[:, :, 0]
        s = local_img[:, :, 1]
        v = local_img[:, :, 2]

        # hd = Hog_descriptor(v ,cell_size=10,bin_size=10)
        # hog_feature_vec, hog_img = hd.extract()
        h_hog_vec = self.hog(h)
        h_hog_vec = np.log(1 + h_hog_vec)  # /500000 #/ 200000.0  # h_hog_vec.max()
        s_hog_vec = self.hog(s)
        s_hog_vec = np.log(1 + s_hog_vec)  # /500000 #/ 200000.0  # s_hog_vec.max()
        v_hog_vec = self.hog(v)
        v_hog_vec = np.log(1 + v_hog_vec)  # /500000 #/ 200000.0  # v_hog_vec.max()
        # print(v_hog_vec.shape)

        color_his_r = cv2.calcHist(cv2.cvtColor(local_img.copy(),
                                                cv2.COLOR_HSV2RGB),
                                   [0],
                                   None,
                                   [64],
                                   [0.0, 256.0])
        color_his_r = color_his_r / 256.0
        color_his_g = cv2.calcHist(cv2.cvtColor(local_img.copy(),
                                                cv2.COLOR_HSV2RGB),
                                   [1],
                                   None,
                                   [64],
                                   [0.0, 256.0])
        color_his_g = color_his_g / 256.0
        color_his_b = cv2.calcHist(cv2.cvtColor(local_img.copy(),
                                                cv2.COLOR_HSV2RGB),
                                   [2],
                                   None,
                                   [64],
                                   [0.0, 256.0])
        color_his_b = color_his_b / 256.0
        # feature_vec = np.zeros([h_hog_vec.shape[0]+s_hog_vec.shape[0]+v_hog_vec.shape[0]+color_his.shape[0]])
        feature_vec = np.concatenate([h_hog_vec,
                                      s_hog_vec,
                                      v_hog_vec,
                                      color_his_r.reshape([-1]),
                                      color_his_g.reshape([-1]),
                                      color_his_b.reshape([-1])])  # ,color_his])
        feature_vec = feature_vec.astype(dtype=np.float)

        return feature_vec

        # print(hog_feature_vec.shape)
        # return hog_feature_vec

    def keyPointProcess(self):
        orb = cv2.ORB_create(5000)

        kp, des = orb.detectAndCompute(self.src_img, None)

        self.key_img = self.src_img.copy()
        cv2.drawKeypoints(self.src_img, kp, self.key_img)

        self.tAddImg('key', self.key_img)

    def pltShow(self, index=0):
        plt.figure(index)
        for i in range(len(self.img_list)):
            # if not 'proc' in self.img_name_list[i] :
            #     continue
            plt.subplot(3, len(self.img_list) / 3 + 1, i + 1)
            plt.imshow(self.img_list[i] / np.mean(self.img_list[i]))
            # plt.colorbar()
            plt.title(self.img_name_list[i])
