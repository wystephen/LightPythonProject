# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 24　22:56
import numpy as np
from skimage import data, util, filters, color
from skimage.morphology import watershed
import matplotlib.pyplot as plt

import cv2

coins = cv2.cvtColor(cv2.imread('image/1.jpg'),cv2.COLOR_RGB2GRAY)
edges = filters.sobel(coins)

grid = util.regular_grid(coins.shape, n_points=468)

seeds = np.zeros(coins.shape, dtype=int)
seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1

w0 = watershed(edges, seeds)
w1 = watershed(edges, seeds, compactness=0.01)

fig, (ax0, ax1) = plt.subplots(1, 2)

ax0.imshow(color.label2rgb(w0, coins))
ax0.set_title('Classical watershed')

ax1.imshow(color.label2rgb(w1, coins))
ax1.set_title('Compact watershed')

plt.show()