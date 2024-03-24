# Dataset

Here's a brief introduction of the dataset and disease that  our MedSAM Adapter support now.



## 2D

### ISIC2016

This is a **2D** dataset of **melanoma** or **nevus** segmentation from dermoscopic images and contains **one** foreground category.  Availabel here: [ISIC Challenge (isic-archive.com)](https://challenge.isic-archive.com/data/#2016)



### REFUGE2 

This is a **2D** dataset of **optic disc** and **optic cup** segmentation over fundus images and contains **two** foreground category. Available here: [Program - Grand Challenge (grand-challenge.org)](https://refuge.grand-challenge.org/)



### LIDC

This is a **2D** dataset of **lung** images and contains **one** foreground category. Available here: [LIDC-IDRI Dataset | Papers With Code](https://paperswithcode.com/dataset/lidc-idri)



### DDTI 

This is a 2D dataset for **thyroid nodule** segmentation and contains **one** foreground category . Available here: [DDTI: Thyroid Ultrasound Images (kaggle.com)](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images)****



### WBC

This is a **2D** dataset of **white blood cell **and contains **two** foreground category. Available here: [zxaoyou/segmentation_WBC: White blood cell (WBC) image datasets (github.com)](https://github.com/zxaoyou/segmentation_WBC)

This dataset contains 2 sub datasets, here we use `Dataset1`. You can change it in `dataset/wbc.py`.



### STARE

This is a **2D** dataset of **retinal blood vessel **and contains **one** foreground category. Available here: https://paperswithcode.com/dataset/stare

Can be appointed by `python train.py -dataset STARE ...`



### Pendal

This is a **2D** dataset of **mandible** and contains **one** foreground category. Available here: https://data.mendeley.com/datasets/hxt48yk462/2. 

Can be appointed by `python train.py -dataset pendal ...`

This dataset contains 2 kind of segmentation labels, in folder `Segmentation1` and `Segmentation2`. Here we use the labels in `Segmentation1` as default. This can be changed in `dataset/pendal.py`.




## 3D

### Brat2021 

This is a **3D** dataset of **brain tumors** that come from the MICCAI23 challenge and contains **three** foreground category. Available here: [MICCAI BRATS - The Multimodal Brain Tumor Segmentation Challenge](http://braintumorsegmentation.org/)



### Kits23

This is a **3D** dataset of **kidney tumors** that come from the MICCAI23 challenge and contains **two** foreground category.  Available here: https://kits-challenge.org/kits21/. This dataset contains 2 kind of segmentation labels, namely aggregated_AND_seg.nii.gz， aggregated_OR_seg.nii.gz， aggregated_MAJ_seg.nii.gz. You can change it in `dataset/kits.py`.

Can be appointed by `python train.py -dataset kits ...`



### Atlas 23

This is a **3D** dataset of **liver tumors** that come from the MICCAI23 challenge and contains **two** foreground category.  Available here: https://atlas-challenge.u-bourgogne.fr/dataset. 

Can be appointed by `python train.py -dataset atlas ...`



### LNQ 23

This is a **3D** dataset of **mediastinal lymph node** that come from the MICCAI23 challenge and contains **one** foreground category.  Available here: https://lnq2023.grand-challenge.org/ .

Can be appointed by `python train.py -dataset lnq ...`



### SegRap

This is a **3D** dataset of **nasopharynx cancer** from the MICCAI23 challenge and contains **53** foreground category. Available here:  https://segrap2023.grand-challenge.org/segrap2023/ 

We use synthesized images`image.nii.gz` for each case in folder`SegRap2023_Training_Set_120cases`. As for the labels, we use the labels in `SegRap2023_Training_Set_120cases_OneHot_Labels\Task001`, you can try different kind of labels in the original dataset as well !

Can be appointed by `python train.py -dataset segrap ...`



### Toothfairy 

This is a **3D** dataset of **inferior alveolar nerve** from the MICCAI23 challenge and contains **one** foreground category.  Available here:  https://toothfairy.grand-challenge.org/ 

Can be appointed by `python train.py -dataset toothfairy ...`



