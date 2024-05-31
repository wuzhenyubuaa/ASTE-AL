# <p align=center>`Pixel Is All You Need: Adversarial Spatio-Temporal Ensembles Active Learning for Salient Object Detection`</p>


## Abstract

Although weakly-supervised techniques can reduce the labeling effort, it is unclear whether a saliency model trained with weakly-supervised data (e.g., point annotation) can achieve the equivalent performance of its fully-supervised version. This paper attempts to answer this unexplored question by proving a hypothesis: there is a point-labeled dataset where saliency models trained on it can achieve equivalent performance when trained on the densely annotated dataset. To prove this conjecture, we proposed a novel yet effective adversarial spatio-temporal ensembles active learning (ASTE-AL). Our contributions are three-fold:  1) Our proposed adversarial attack triggering uncertainty can conquer the overconfidence of existing active learning methods and accurately locate these uncertain pixels. 2) Our proposed spatio-temporal ensembles strategy not only achieves better performance than the traditional deep ensembles but significantly reducing the computational cost. 3) Our proposed relationship-aware diversity sampling algorithm can conquer oversampling while boosting model performance.
Experimental results show that our ASTE-AL can find such a point-labeled dataset, where a saliency model trained on it obtained 98\%-99\% performance of its fully-supervised version with only ten annotated points per image. 

## Framework Overview



![arch](README.assets/pipeline.png)

## Result

+ Comparison with the previous state-of-the-art methods with different training sets:

![image-20220601123106208](README.assets/results.png)



## Usage

## Prerequisites
- [Python 3.5](https://www.python.org/)
- [Pytorch 1.3](http://pytorch.org/)
- [OpenCV 4.0](https://opencv.org/)
- [Numpy 1.15](https://numpy.org/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [Apex](https://github.com/NVIDIA/apex)


## Clone repository

```shell
git clone git@github.com:weijun88/F3Net.git
cd F3Net/
```

## Download dataset

Download the following datasets and unzip them into `data` folder

- [PASCAL-S](http://cbi.gatech.edu/salobj/)
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [DUTS](http://saliencydetection.net/duts/)

4. **Update the paths**

    Replace all `/root/datasets/sod/GCoNet_plus` and `/root/codes/sod/GCoNet_plus` in this project to  `/YOUR_PATH/datasets/sod/GCoNet_plus` and `/YOUR_PATH/codes/sod/GCoNet_plus`, respectively.

5. **Training + Test + Evaluate + Select the Best**

    `./gco.sh`

    If you can apply more GPUs on the DGX cluster, you can `./sub_by_id.sh` to submit multiple times for more stable results.

    If you have the OOM problem, plz decrease `batch_size` in `config.py`.

6. **Adapt the settings of modules in config.py**

    You can change the weights of losses, try various *backbones* or use different *data augmentation* strategies. There are also some modules coded but not used in this work, like *adversarial training*, the *refiner* in [BASNet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf), weighted *multiple output and supervision* used in [GCoNet](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Group_Collaborative_Learning_for_Co-Salient_Object_Detection_CVPR_2021_paper.pdf) and [GICD](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570443.pdf), etc.

    ![image-20220426234911555](README.assets/config.png)

## Download

​	Find **well-trained models** + **predicted saliency maps** and all other stuff on my [google-drive folder for this work](https://drive.google.com/drive/folders/1SIr_wKT3MkZLtZ0jacOOZ_Y5xnl9-OPw?usp=sharing):

![GD_content](README.assets/GD_content.png)

## Acknowledgement

We appreciate the codebases of [GICD](https://github.com/zzhanghub/gicd), [GCoNet](https://github.com/fanq15/GCoNet). Thanks for the CoSOD evaluation toolbox provided in [eval-co-sod](https://github.com/zzhanghub/eval-co-sod). Thanks for the drawing codes of figure 1 from [DGNet](https://github.com/GewelsJI/DGNet).

### Citation

```
@article{zheng2022gconet+,
  author={Zheng, Peng and Fu, Huazhu and Fan, Deng-Ping and Fan, Qi and Qin, Jie and Tai, Yu-Wing and Tang, Chi-Keung and Van Gool, Luc},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={GCoNet+: A Stronger Group Collaborative Co-Salient Object Detector}, 
  year={2023},
  volume={45},
  number={9},
  pages={10929-10946},
  doi={10.1109/TPAMI.2023.3264571}
}
@inproceedings{fan2021gconet,
  title = {Group Collaborative Learning for Co-Salient Object Detection},
  author = {Fan, Qi and Fan, Deng-Ping and Fu, Huazhu and Tang, Chi-Keung and Shao, Ling and Tai, Yu-Wing},
  booktitle = CVPR,
  year = {2021}
}
```



## Contact

Any question, discussion or even complaint, feel free to leave issues here or send me e-mails (zhengpeng0108@gmail.com).

