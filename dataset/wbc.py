import glob
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click


class WBC(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        self.data_path = os.path.join(data_path,'Dataset1')
        self.name_list = glob.glob(self.data_path + "/*.bmp")
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1     # available: 1 2

        """Get the images"""
        name = os.path.basename(self.name_list[index]).split('.')[0]

        img_path = os.path.join(self.data_path, name + '.bmp')
        msk_path = os.path.join(self.data_path, name + '.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        mask = np.array(mask) // 127
        mask[mask!=point_label] = 0
        mask[mask==point_label] = 255

        if self.prompt == 'click':
            point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)

            if self.transform_msk:
                mask = Image.fromarray(mask)
                mask = self.transform_msk(mask).int()
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }