# -*- coding:utf-8 -*-
# carete by steve at  2018 / 03 / 03　13:44

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

class imuread:
    def __init__(self,file_name = 'MT_07700791-002-000.csv'):
        self.file_name = file_name

    def load(self):
        file_lines = open(self.file_name).readlines()

        self.data = np.zeros([len(file_lines)-7,14])

        for i in range(7,len(file_lines)):
            print(file_lines[i])
            for unit in file_lines[i].split('\t'):
                print(unit,',')

if __name__ == '__main__':
    ir = imuread()

    ir.load()