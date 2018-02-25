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
        scale_to_width = 600
        if img.shape[0] > scale_to_width:
            scale = int(img.shape[0] / scale_to_width)

        # self.src_img = img
        self.src_img = cv2.resize(img,
                                  (int(img.shape[0] / scale), int(img.shape[1] / scale)),
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

    def multiLayerProcess(self, win_size_list=[100, 200, 400]):
        self.result_imgs = list()

        for win_size in win_size_list:
            tmp_res_img = self.original_img.copy()
            for i in range(0, self.original_img.shape[0], win_size):
                for j in range(0, self.original_img.shape[1], win_size):
                    end_x = min(i + win_size, self.original_img.shape[0])
                    end_y = min(j + win_size, self.original_img.shape[1])
                    tmp_res_img[i:end_x, j:end_y, :] = \
                        self.sub_image_process(self.original_img[i:end_x, j:end_y, :])
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
        r = self.sub_img[:, :, 0]
        g = self.sub_img[:, :, 1]
        b = self.sub_img[:, :, 2]



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
        color_mask  =  h

        if h_count + v_count > 50 \
                and h_count < 0.7 * (h_count + v_count) \
                and v_count < 0.7 * (h_count + v_count):
            color_mask = np.zeros_like(color_mask)
        else:
            color_mask = np.ones_like(color_mask) * 255

        self.h_line = np.zeros_like(self.h_canny)
        self.s_line = np.zeros_like(self.s_canny)
        self.v_line = np.zeros_like(self.v_canny)
        self.h_line = self.detectAnddraw(self.h_canny)
        self.s_line = self.detectAnddraw(self.s_canny)
        self.v_line = self.detectAnddraw(self.v_canny)



        # self.sub_img = cv2.addWeighted(cv2.cvtColor(color_mask, cv2.COLOR_GRAY2RGB)
        #                                , 0.5, self.sub_img, 0.5, 0)
        self.sub_img[:,:,0] = self.h_canny
        self.sub_img[:,:,0] = self.h_line
        self.sub_img[:,:,1] = self.s_canny
        self.sub_img[:,:,1] = self.s_line
        self.sub_img[:,:,2] = self.v_canny
        self.sub_img[:,:,2] = self.v_line

        # self.sub_img[:,:,:1] =  0
        cv2.rectangle(self.sub_img, (0, 0), (height, width), (0, 0, 223), 2)

        return cv2.resize(self.sub_img, (width, height))

    def detectAnddraw(self,img):

        # 经验参数
        minLineLength = 50
        maxLineGap = 5
        # 直线检测，（电线杆上有直线，自然界直线比较少）
        line_list = (transform.probabilistic_hough_line(img, threshold=30,
                                                    line_length=minLineLength,
                                                    line_gap=maxLineGap))
        lines = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
        if(len(line_list)<15):
            for (x1, y1), (x2, y2) in line_list:
                cv2.line(lines, (x1, y1), (x2, y2), (255, 255, 255), 2)

        return cv2.cvtColor(lines,cv2.COLOR_RGB2GRAY)

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

    def pltShow(self, index=0):
        plt.figure(index)
        for i in range(len(self.img_list)):
            # if not 'proc' in self.img_name_list[i] :
            #     continue
            plt.subplot(3, len(self.img_list) / 3 + 1, i + 1)
            plt.imshow(self.img_list[i] / np.mean(self.img_list[i]))
            # plt.colorbar()
            plt.title(self.img_name_list[i])
