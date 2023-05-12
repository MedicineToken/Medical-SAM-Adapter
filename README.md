# Medical-SAM-Adapter

Medical SAM Adapter, or say MSA, is a project to fineturn [SAM](https://github.com/facebookresearch/segment-anything) using [Adaption](https://lightning.ai/pages/community/tutorial/lora-llm/) for the Medical Imaging.
This method is elaborated in the paper [Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation](https://arxiv.org/abs/2304.12620).


## A Quick Overview 

<img width="880" height="380" src="https://github.com/WuJunde/Medical-SAM-Adapter/blob/main/figs/medsamadpt.jpeg">

## News
- 23-05-10. This project is still quickly updating 🌝. Check TODO list to see what will be released next.
- 23-05-11. GitHub Dicussion opened. You guys can now talk, code and make friends on the playground 👨‍❤️‍👨. 

## Requirement

``conda env create -f environment.yml``

Download [SAM checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth), and put it at ./checkpoint/sam/

## Example Cases
### Melanoma Segmentation from Skin Images (2D)

1. Download ISIC dataset part 1 from https://challenge.isic-archive.com/data/. Then put the csv files in "./data/isic" under your data path. Your dataset folder under "your_data_path" should be like:

ISIC/

     ISBI2016_ISIC_Part1_Test_Data/...
     
     ISBI2016_ISIC_Part1_Training_Data/...
     
     ISBI2016_ISIC_Part1_Test_GroundTruth.csv
     
     ISBI2016_ISIC_Part1_Training_GroundTruth.csv
    
2. Begin Adapting! run: ``python train.py -net sam -mod sam_adpt -exp_name *msa_test_isic* -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 32 -dataset isic --data_path *../data*``
change "data_path" and "exp_name" for your own useage. 

3. Evaluation: The code can automatically evaluate the model on the test set during traing, set "--val_freq" to control how many epoches you want to evaluate once. You can also run val.py for the independent evaluation.

4. Result Visualization: You can set "--vis" parameter to control how many epoches you want to see the results in the training or evaluation process.

In default, everything will be saved at `` ./logs/`` 

### Abdominal Multiple Organs Segmentation (3D)

This tutorial demonstrates how MSA can adapt SAM to 3D multi-organ segmentation task using the BTCV challenge dataset.

For BTCV dataset, under Institutional Review Board (IRB) supervision, 50 abdomen CT scans of were randomly selected from a combination of an ongoing colorectal cancer chemotherapy trial, and a retrospective ventral hernia study. The 50 scans were captured during portal venous contrast phase with variable volume sizes (512 x 512 x 85 - 512 x 512 x 198) and field of views (approx. 280 x 280 x 280 mm3 - 500 x 500 x 650 mm3). The in-plane resolution varies from 0.54 x 0.54 mm2 to 0.98 x 0.98 mm2, while the slice thickness ranges from 2.5 mm to 5.0 mm.

Target: 13 abdominal organs including
Spleen
Right Kidney
Left Kidney
Gallbladder
Esophagus
Liver
Stomach
Aorta
IVC
Portal and Splenic Veins
Pancreas
Right adrenal gland
Left adrenal gland.
Modality: CT
Size: 30 3D volumes (24 Training + 6 Testing)
Challenge: BTCV MICCAI Challenge
The following figure shows image patches with the organ sub-regions that are annotated in the CT (top left) and the final labels for the whole dataset (right).


1. Prepare BTCV dataset following [MONAI](https://docs.monai.io/en/stable/index.html) instruction:

Download BTCV dataset from: https://www.synapse.org/#!Synapse:syn3193805/wiki/217752. After you open the link, navigate to the "Files" tab, then download Abdomen/RawData.zip.

After downloading the zip file, unzip. Then put images from RawData/Training/img in ../data/imagesTr, and put labels from RawData/Training/label in ../data/labelsTr.

Download the json file for data splits from this [link](https://drive.google.com/file/d/1qcGh41p-rI3H_sQ0JwOAhNiQSXriQqGi/view). Place the JSON file at ../data/dataset_0.json.

2. For the Adaptation, run: ``python train.py -net sam -mod sam_adpt -exp_name msa-3d-sam-btcv -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 8 -dataset decathlon -thd True -chunk 96 -dataset ../data -num_sample 4``  

You can modify following parameters to save the memory usage: '-b' the batch size, '-chunk' the 3D depth (channel) for each sample, '-num_sample' number of samples for [Monai.RandCropByPosNegLabeld](https://docs.monai.io/en/stable/transforms.html#randcropbyposneglabeld), 'evl_chunk' the 3D channel split step in the evaluation, decrease it if out of memory in the evaluation. 

## Run on  your own dataset
It is simple to run MSA on the other datasets. Just write another dataset class following which in `` ./dataset.py``. You only need to make sure you return a dict with 


     {
                 'image': A tensor saving images with size [C,H,W] for 2D image, size [C, H, W, D] for 3D data.
                 D is the depth of 3D volume, C is the channel of a scan/frame, which is commonly 1 for CT, MRI, US data. 
                 If processing, say like a colorful surgical video, D could the number of time frames, and C will be 3 for a RGB frame.

                 'label': The target masks. Same size with the images except the resolutions (H and W).

                 'p_label': The prompt label to decide positive/negative prompt. To simplify, you can always set 1 if don't need the negative prompt function.

                 'pt': The prompt. Should be the same as that in SAM, e.g., a click prompt should be [x of click, y of click], one click for each scan/frame if using 3d data.

                 'image_meta_dict': Optional. if you want save/visulize the result, you should put the name of the image in it with the key ['filename_or_obj'].

                 ...(others as you want)
     }


Welcome to open issues if you meet any problem. It would be appreciated if you could contribute your dataset extensions. Unlike natural images, medical images vary a lot depending on different tasks. Expanding the generalization of a method requires everyone's efforts.

### TODO LIST

- [ ] Jupyter tutorials.
- [x] Fix bugs in BTCV. Add BTCV example.
- [ ] Release REFUGE2, BraTs dataloaders and examples
- [ ] Changable Image Resolution 
- [ ] Fix bugs in Multi-GPU parallel
- [x] Sample and Vis in training
- [ ] Release general data pre-processing and post-processing
- [x] Release evaluation
- [ ] Deploy on HuggingFace
- [x] configuration
- [ ] Release SSL code

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



