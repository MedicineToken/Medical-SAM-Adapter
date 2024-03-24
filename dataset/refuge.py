import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click


class REFUGE(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'none', plane = False):
        self.data_path = data_path
        self.subfolders = [f.path for f in os.scandir(os.path.join(data_path, mode + '-400')) if f.is_dir()]
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):
        point_label = 1

        """Get the images"""
        subfolder = self.subfolders[index]
        name = subfolder.split('/')[-1]

        # raw image and raters path
        img_path = os.path.join(subfolder, name + '.jpg')
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]
        multi_rater_disc_path = [os.path.join(subfolder, name + '_seg_disc_' + str(i) + '.png') for i in range(1, 8)]

        # raw image and raters images
        img = Image.open(img_path).convert('RGB')
        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]
        multi_rater_disc = [Image.open(path).convert('L') for path in multi_rater_disc_path]

        # resize raters images for generating initial point click
        newsize = (self.img_size, self.img_size)
        multi_rater_cup_np = [np.array(single_rater.resize(newsize)) for single_rater in multi_rater_cup]
        multi_rater_disc_np = [np.array(single_rater.resize(newsize)) for single_rater in multi_rater_disc]

        # first click is the target agreement among most raters
        if self.prompt == 'click':
            point_label, pt = random_click(np.array(np.mean(np.stack(multi_rater_cup_np), axis=0)) / 255, point_label)
            point_label, pt_disc = random_click(np.array(np.mean(np.stack(multi_rater_disc_np), axis=0)) / 255, point_label)
        else:
            # you may want to get rid of click prompts
            pt = np.array([0, 0], dtype=np.int32)
            
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)
            # transform to mask size (out_size) for mask define
            mask_cup = F.interpolate(multi_rater_cup, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)

            multi_rater_disc = [torch.as_tensor((self.transform(single_rater) >0.5).float(), dtype=torch.float32) for single_rater in multi_rater_disc]
            multi_rater_disc = torch.stack(multi_rater_disc, dim=0)
            mask_disc = F.interpolate(multi_rater_disc, size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0)
            torch.set_rng_state(state)
            
            mask = torch.concat([mask_cup, mask_disc], dim=0)

        if self.prompt == 'box':
            x_min_cup, x_max_cup, y_min_cup, y_max_cup = random_box(multi_rater_cup)
            box_cup = [x_min_cup, x_max_cup, y_min_cup, y_max_cup]
            x_min_disc, x_max_disc, y_min_disc, y_max_disc = random_box(multi_rater_disc)
            box_disc = [x_min_disc, x_max_disc, y_min_disc, y_max_disc]
        else:
            # you may want to get rid of box prompts
            box_cup = [0, 0, 0, 0]
            box_disc = [0, 0, 0, 0]

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'box': box_cup,
            'image_meta_dict':image_meta_dict,
        }