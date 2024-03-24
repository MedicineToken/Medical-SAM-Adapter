import os
import pickle

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils import generate_click_prompt, random_box, random_click


class SegRap(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        self.args = args
        self.data_path = data_path
        self.name_list = os.listdir(os.path.join(self.data_path,'SegRap2023_Training_Set_120cases_OneHot_Labels','Task001'))
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1
        label = 1 # 待分割的类别

        """Get the images"""
        name = self.name_list[index].split('.')[0]
        img = nib.load(os.path.join(self.data_path,'SegRap2023_Training_Set_120cases',name,'image.nii.gz')).get_fdata()
        mask = nib.load(os.path.join(self.data_path,'SegRap2023_Training_Set_120cases_OneHot_Labels','Task001',name+'.nii.gz')).get_fdata()

        img = np.resize(img,(self.args.image_size, self.args.image_size,img.shape[-1]))
        mask = np.resize(mask,(self.args.out_size,self.args.out_size,mask.shape[-1]))
        mask[mask!=label] = 0
        mask[mask==label] = 1
        
        img = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0).int()
        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask), point_label)

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }

