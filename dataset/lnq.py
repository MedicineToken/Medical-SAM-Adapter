import json
import os
import pickle

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils import generate_click_prompt, random_box, random_click


class LNQ(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):


        self.args = args
        self.data_path = os.path.join(data_path,'train')

        files = os.listdir(self.data_path)

        self.name_list = [file for file in files if file.endswith('.png')]
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)


    def __getitem__(self, index):
        point_label = 1
        label = 1 

        """Get the images"""
        name = self.name_list[index].split('.')[0]
        img_name = name + '-ct.nrrd'
        mask_name = name + '-seg.nrrd'

        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_path,img_name)))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_path,mask_name)))

        mask[mask!=label] = 0
        mask[mask==label] = 1
        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])
        img = np.transpose(img,(1,2,0))
        mask = np.transpose(mask,(1,2,0))

        # img = np.resize(mask,(self.args.image_size, self.args.image_size,128))
        # mask = np.resize(mask,(self.args.out_size,self.args.out_size,128))

        # # img = np.resize(img,(self.args.image_size, self.args.image_size,img.shape[-1]))
        # # mask = np.resize(mask,(self.args.out_size,self.args.out_size,mask.shape[-1]))

        img = torch.tensor(img).unsqueeze(0).int()
        mask = torch.tensor(mask).unsqueeze(0).int()

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask), point_label)

        name = img_name
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }

