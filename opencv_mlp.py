# -*- coding:utf-8 -*-
# carete by steve at  2018 / 02 / 25　20:34

import cv2
import os
if __name__ == '__main__':
    # print(os.listdir())
    # dnn_model = cv2.dnn.readNetFromTensorflow(model='./model/inception/classify_image_graph_def.pb',
    #                                           config='./model/inception/imagenet_2012_challenge_label_map_proto.pbtxt')
    # dnn_model = cv2.dnn.readNetFromTensorflow('./model/ssd_mobilenet/saved_model.pb')
    dnn_model = cv2.dnn.readNetFromDarknet(cfgFile='./modle/yolo/yolo.cfg',
                                           darknetModel='./model/yolo/yolo.weights')
    print(dnn_model)
