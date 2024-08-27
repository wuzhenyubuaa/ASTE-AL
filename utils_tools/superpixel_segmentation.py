import os
import cv2
import numpy as np

path = '/media/jack/ZhenyuWu/mywork/dataset/sod/dut_tr/img/'
img_list = os.listdir(path)
area_list = []

for name in img_list:
    image = cv2.imread(path  + name )
    seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1], image.shape[0], image.shape[2], 1000, 15, 3, 5, True)
    seeds.iterate(image, 20)  # The input image size must be the same as the initialization shape, and the number of iterations is 10.
    mask_seeds = seeds.getLabelContourMask()
    label_seeds = seeds.getLabels()
    number_seeds = seeds.getNumberOfSuperpixels()
    # label_flatten = label_seeds.flatten()
    # sum = 0
    # for i in range(0, number_seeds):
    #     sum += np.where(label_seeds==0)[0].shape
    area_list.append(number_seeds)


