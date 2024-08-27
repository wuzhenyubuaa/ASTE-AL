import pickle as pkl
import numpy as np
import cv2
import matplotlib.pyplot as plt

# This file can be used to generate Figure 11
label_path ='/userhome/wuzhenyu/mywork/work7/PointSOD/experiments/model-7/query_0/label.pkl'
label = pkl.load(open(label_path, 'rb'))
distri= np.zeros((352, 352))
for i in range(0, len(label)):
    rs = cv2.resize(label[0].astype('uint8'),(352, 352))
    distri += rs
np.save('./point_dis.npy', distri)
plt.matshow(distri, cmap='viridis')
plt.axis('off')
plt.savefig('./distri.png',bbox_inches='tight')