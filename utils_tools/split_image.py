import cv2

root = '/home/jack/mywork/work7/writer_paper/pipeline_pic/5/'
img = cv2.imread('/home/jack/mywork/work7/PointSOD/experiments/model-2/query_0/epoch_15/ILSVRC2012_test_00000172.png')

image = img[:,:176,:]

gt = img[:,176:352,:]

pred = img[:,352:528,0]

org = img[:,528:704,0]

ent = img[:, 880:1056,0]

uncertainty = img[:, 528:1056,0]

image = cv2.resize(image,(352,352),cv2.INTER_NEAREST)
pred = cv2.resize(pred,(352,352), cv2.INTER_NEAREST)
gt = cv2.resize(gt,(352,352), cv2.INTER_NEAREST)
ent = cv2.resize(ent,(352,352), cv2.INTER_NEAREST)

errors = gt[:,:,0] -pred

cv2.imwrite(root+ 'image.png', image)
cv2.imwrite(root+ 'gt.png', gt)
cv2.imwrite(root+ 'pred.png', pred)

cv2.imwrite(root+ 'org.png', org)
cv2.imwrite(root+ 'ent.png', ent)
cv2.imwrite(root+ 'errors.png', errors)
cv2.imwrite(root+ 'uncertainty.png', uncertainty)
print(img.shape)