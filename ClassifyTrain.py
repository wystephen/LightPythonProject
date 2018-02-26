# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 26　11:58
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

# from sklearn import

from sklearn.metrics import *
from sklearn.externals import joblib

if __name__ == '__main__':
    x = np.loadtxt('data_x.csv', delimiter=',')
    y = np.loadtxt('data_y.csv', delimiter=',')

    print(x.shape, y.shape)

    print(len(np.where(y > 0.5)[0]))
    weight = np.ones_like(y)
    weight[np.where(y < 0.5)] = 1.0
    weight[np.where(y> 0.5)] = 1.0

    # clf = svm.NuSVC(kernel='rbf')
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=3,random_state=0)

    clf.fit(x.copy(), y.copy(), sample_weight=weight)

    pre_y = clf.predict(x.copy())

    # precision, recall, thresholds = precision_recall_curve(y,pre_y)
    # fpr,tpr,thresholds = roc_curve(y,pre_y)
    print(accuracy_score(y.copy(), pre_y, sample_weight=weight))

    print('error num:', np.sum(np.abs(y - pre_y)))
    # plt.figure()
    # plt.plot(fpr,tpr)
    joblib.dump(clf, 'model/svm/train_model.m')
    # plt.figure()
    # plt.plot(y, label='y')
    # plt.plot(pre_y, label='pre_y')
    # plt.plot(np.abs(y-pre_y))
    # plt.legend()

    # plt.figure()
    # plt.imshow(x)
    # plt.colorbar()
    # plt.show()
