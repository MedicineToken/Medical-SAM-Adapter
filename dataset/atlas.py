import json
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


class Atlas(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):


        self.args = args
        self.data_path = os.path.join(data_path,'train')
        with open(os.path.join(self.data_path,'dataset.json'),'r') as file:
            data = json.load(file)
        self.name_list = data['training']
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
        img_name = self.name_list[index]['image']
        mask_name = self.name_list[index]['label']

        img = nib.load(os.path.join(self.data_path,img_name)).get_fdata()
        mask = nib.load(os.path.join(self.data_path,mask_name)).get_fdata()

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

        img = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask), point_label)
        # if self.transform:
        #     state = torch.get_rng_state()
        #     img = self.transform(img)
        #     torch.set_rng_state(state)

        #     if self.transform_msk:
        #         mask = self.transform_msk(mask)
                
        #     # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
        #     #     mask = 1 - mask
        name = img_name
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }

