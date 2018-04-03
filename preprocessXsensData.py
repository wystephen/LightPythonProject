# -*- coding:utf-8 -*-
# carete by steve at  2018 / 03 / 08　15:14

import numpy  as np
import scipy as sp
import matplotlib.pyplot as plt

from ImuData.ImuPreProcess import *

import os

if __name__ == '__main__':
    dir_name = 'XsensData\\'

    for file_name in os.listdir(dir_name):
        if 'CVS' in file_name:
            ir = imuread(dir_name+file_name)
            ir.load()
            ir.save(dir_name+file_name.split('.')[0]+'.csv')



