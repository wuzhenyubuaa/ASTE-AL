import cv2
import os
import numpy  as np
import matplotlib.pyplot as plt
envpath = '/home/jack/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


syn_data = '/media/jack/ZhenyuWu/mywork/dataset/sod/dut_tr/gt/'


# 1.  物体位置分布
# syn_list = os.listdir(syn_data)
# gt_sum = np.zeros([352,352])
# for data in syn_list:
#
#     gt = cv2.imread(syn_data+data, 0)
#     gt = cv2.resize(gt, (352, 352))
#
#     gt_sum += gt
#     print(gt.shape)
#
# gt_sum = gt_sum / len(syn_list)


distri = np.load('/home/jack/mywork/work7/writer_paper/point_dis.npy')
distri =  ((distri/10553) * 255)
plt.matshow(distri.astype('uint8'), cmap='viridis')
plt.axis('off')
plt.savefig('./distri.png',bbox_inches='tight')

# plt.matshow(gt_sum, cmap='viridis')
# plt.axis('off')
# plt.savefig('./444.png',bbox_inches='tight')


# 2. 物体大小分布
# dut_omron = np.load('/media/jack/新加卷/mywork/work6/ResNeSt-master/dut_omron_hist.npy')
# print(dut_omron.shape)


# 3 instance size
#
# syn_list = os.listdir(syn_data)
# size_list =[]
# for name in syn_list:
#
#     gt = cv2.imread(syn_data+name, 0)
#     (h, w) = gt.shape
#     size = (np.where(gt>0)[0].shape[0]) / (h*w)
#
#     size_list.append(size)
#
#
#     print(gt.shape)