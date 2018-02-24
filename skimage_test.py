# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 24　22:50
import numpy as np
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
import cv2

image = cv2.cvtColor(cv2.cvtColor(cv2.imread('image/1.jpg'),
                                  cv2.COLOR_RGB2HSV),
                     cv2.COLOR_RGB2GRAY)
edge_roberts = roberts(image)
edge_sobel = sobel(image)

fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
                       figsize=(8, 4))

ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
ax[0].set_title('Roberts Edge Detection')

ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
ax[1].set_title('Sobel Edge Detection')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
