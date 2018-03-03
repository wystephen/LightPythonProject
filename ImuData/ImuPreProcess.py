# -*- coding:utf-8 -*-
# carete by steve at  2018 / 03 / 03　13:44

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import re
import time

import datetime
class imuread:
    def __init__(self,file_name = 'MT_07700791-003-000.csv'):
        self.file_name = file_name

    def load(self):
        file_lines = open(self.file_name).readlines()

        self.data = np.zeros([len(file_lines)-7,10])

        for i in range(7,len(file_lines)):
            # print(file_lines[i])
            matcher = re.compile('[-]{0,1}[0-9]{1,3}\.{0,1}[0-9]{0,15}')
            all_num = matcher.findall(file_lines[i])

            # print(tt)
            tt = datetime.datetime(int(all_num[2]),int(all_num[3]),int(all_num[4]),int(all_num[5]),int(all_num[6]),int(all_num[7]))

            print(tt.timestamp()+float(all_num[1])*1e-9)
            self.data[i-7,0] = tt.timestamp()+float(all_num[0])*1e-9


            # print(all_num)
            for j in range(9):
                self.data[i-7, 1+j] = float(all_num[j+len(all_num)-9])

        plt.figure()
        # plt.imshow(self.data/self.data.std(axis=1))
        plt.imshow(self.data)
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    ir = imuread(file_name='2018-03-03-17h35.TXT')

    ir.load()