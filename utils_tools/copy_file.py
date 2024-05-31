import shutil
import os


img_path = '/media/jack/新加卷/data/sod/dut_omron/img/';
gt_path = '/media/jack/新加卷/data/sod/dut_omron/gt/';



our_10_path = '/home/jack/mywork/work7/PointSOD/experiments/model-1/query_0/self_training/dut_omron/';
our_20_path = '/home/jack/mywork/work7/PointSOD/experiments/model-1/query_1/self_training/dut_omron/';




F3Net20_path  = '/media/jack/新加卷/mywork/SOTA_SOD_saliency_maps/F3Net2020/dut_omron/';
MINet2020_path = '/media/jack/新加卷/mywork/SOTA_SOD_saliency_maps/MINet2020/dut_omron/';

MWS19_path = '/media/jack/新加卷/mywork/SOTA_SOD_saliency_maps/weakly_supervised/mws_results/dut_omron/';

WSSSA_path = '/media/jack/新加卷/mywork/SOTA_SOD_saliency_maps/weakly_supervised/scribble_cvpr2020/dut_omron/';



EDNS20_path = '/media/jack/新加卷/mywork/SOTA_SOD_saliency_maps/weakly_supervised/EDNS-ECCV2020/dut_omron/';
SCWSSOD_path = '/media/jack/新加卷/mywork/SOTA_SOD_saliency_maps/weakly_supervised/scwssod_aaai2021/dut_omron/';
FCSOD21_path = '/media/jack/新加卷/mywork/SOTA_SOD_saliency_maps/semi_supervised/FCSOD/dut_omron/';

MFNet21_path = '/media/jack/ZhenyuWu/mywork/SOTA-saliency-maps/MFNet_ICCV2021_saliency maps/resnet50-salmap/dut_omron/';


# img_list = ['0133.png', '0138.png', '0163.png', '0172.png', '0173.png', '0217.png', '0233.png', '0284.png']  # hku
img_list = ['sun_aaadbvospbcqnnvq.png', 'sun_aacvyknwkfzsxjbt.png', 'sun_aafvrsnoikyyntky.png', 'sun_aaskgwbtstvsjhxz.png',
            'sun_aaujfkmqhxlcvsbj.png', 'sun_abbylxxljmhcnxql.png', 'sun_abkqvtadghmyaozq.png']

root_path = '/home/jack/mywork/work7/writer_paper/main_visual_comparisons/'

for img in img_list:
    dst_path = root_path + img[:-4]
    os.makedirs(dst_path, exist_ok=True)

    shutil.copy(img_path+img[:-4]+'.jpg', dst_path +'/img'+'.jpg' )
    shutil.copy(gt_path + img, dst_path +  '/gt.png')

    shutil.copy(F3Net20_path + img, dst_path + '/F3Net20.png')

    shutil.copy(our_10_path + img, dst_path + '/our_10.png')
    shutil.copy(our_20_path + img, dst_path + '/our_20.png')

    shutil.copy(MFNet21_path + img, dst_path + '/MFNet21.png')
    shutil.copy(MWS19_path + img, dst_path + '/MWS19.png')
    shutil.copy(EDNS20_path + img, dst_path + '/ENDS20.png')

    shutil.copy(WSSSA_path + img, dst_path + '/WS3A20.png')

    shutil.copy(FCSOD21_path + img, dst_path + '/FCSOD20.png')


    shutil.copy(SCWSSOD_path + img, dst_path + '/SCWSSOD21.png')



