# Medical-SAM-Adapter

Medical SAM Adapter, or say MSA, is a project to fineturn [SAM](https://github.com/facebookresearch/segment-anything) using [Adaption](https://huggingface.co/blog/lora) for the Medical Imaging usage.
This method is elaborated in the paper [Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation](https://arxiv.org/abs/2304.12620).


## A Quick Overview 

|<img width="880" height="170" src="https://github.com/WuJunde/Medical-SAM-Adapter/blob/master/figs/medsamadpt.jpeg">
| **Medical-SAM-Adapter** |

## News
- 22-11-30. This project is still quickly updating. Check TODO list to see what will be released next.

## Requirement

``conda env create -f environment.yml``

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), and put it at ./checkpoint/sam/

## Example Cases
### Melanoma Segmentation from Skin Images (2D)
1. Download ISIC dataset from https://challenge.isic-archive.com/data/. Your dataset folder under "your_data_path" should be like:

ISIC/

     ISBI2016_ISIC_Part3B_Test_Data/...
     
     ISBI2016_ISIC_Part3B_Training_Data/...
     
     ISBI2016_ISIC_Part3B_Test_GroundTruth.csv
     
     ISBI2016_ISIC_Part3B_Training_GroundTruth.csv
    
2. For training, run: ``python train.py -net sam -mod sam_adpt -exp_name *msa_test_isic* -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 32 -dataset isic --data_path *../data*``

change "data_path" and "exp_name" for your own useage. 

The code can automatically evaluate the model on the test set during traing, set "--val_freq" to control how many epoches you want to evaluate once.
    
You can also set "--vis" parameter to visualize the results.

In default, everything will be saved at `` ./logs/`` 

### Abdominal Multiple Organs Segmentation (3D)

...to be continue

## Cite
Please cite
~~~
@article{wu2023medical,
  title={Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation},
  author={Wu, Junde and Fu, Rao and Fang, Huihui and Liu, Yuanpei and Wang, Zhaowei and Xu, Yanwu and Jin, Yueming and Arbel, Tal},
  journal={arXiv preprint arXiv:2304.12620},
  year={2023}
}
~~~



