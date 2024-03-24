import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils import generate_click_prompt, random_box, random_click


class Brat(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        self.args = args
        self.data_path = os.path.join(data_path,'Data')
        self.name_list = os.listdir(self.data_path)
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def load_all_levels(self,path):
        import nibabel as nib
        data_dir = os.path.join(self.data_path)
        levels = ['t1','flair','t2','t1ce']
        raw_image = [nib.load(os.path.join
        (data_dir,path,path+'_'+level+'.nii.gz')).get_fdata() for level in levels]
        raw_seg = nib.load(os.path.join(data_dir,path,path+'_seg.nii.gz')).get_fdata()

        return raw_image[0], raw_seg

    def __getitem__(self, index):
        # if self.mode == 'Training':
        #     point_label = random.randint(0, 1)
        #     inout = random.randint(0, 1)
        # else:
        #     inout = 1
        #     point_label = 1
        point_label = 1
        label = 4   # the class to be segmented

        """Get the images"""
        name = self.name_list[index]
        img,mask = self.load_all_levels(name)

        mask[mask!=label] = 0
        mask[mask==label] = 1
        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        
        img = np.resize(img,(self.args.image_size, self.args.image_size,img.shape[-1]))
        mask = np.resize(mask,(self.args.out_size,self.args.out_size,mask.shape[-1]))

        img = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(mask).unsqueeze(0)
        mask = torch.clamp(mask,min=0,max=1).int()

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
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }

