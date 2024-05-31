import os
import shutil

pred_path = '/media/jack/ZhenyuWu/mywork/SOTA-saliency-maps/PSGLoss_TIP2021/model&results/hku/'

gt_path = '/media/jack/新加卷/data/sod/hku/gt/'

dst_path = '/media/jack/新加卷/data/sod/hku/gt1447/'

dst = '/media/jack/ZhenyuWu/mywork/SOTA-saliency-maps/SAL_WACV_2022/WACV_2022_SAL_PRED/'
gt_file = os.listdir(gt_path)

pred_file = os.listdir(pred_path)


path = '/media/jack/ZhenyuWu/mywork/work6/write_paper/biggan_picture/'

src_path = '/media/jack/ZhenyuWu/mywork/work6/SODGAN/saved_model/biggan/all/model-1/samples_10k/image/'
with open(path + 'test', 'r') as lines:
    samples = []
    for line in lines:
        samples.append(line.strip())
        print(line.strip())
        shutil.copy(src_path+ line.strip()[:-4] +'.jpg', path+'展示的图/image/'+ line.strip()[:-4] + '.jpg')

# for f in gt_file:
#
#     if f not in pred_file:
#
#         # shutil.move(pred_path+f, dst+f)
#
#
#         shutil.move(gt_path+f, dst_path+f)
#
#     # os.rename(pred_path+f, pred_path+f[:-8]+'.png')


