import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click


class DDTI(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):

        self.name_list = os.listdir(os.path.join(data_path,mode,'images'))
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.name_list)

    def find_connected_components(self,mask):
        mask = np.clip(mask,0,1)
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        point = []
        point_labels = []

        for label in range(1, num_labels):
            component_mask = np.where(labels == label, 1, 0)
            area = np.sum(component_mask)

            if area > 400:
                point_label, random_point = random_click(component_mask)
                point.append(random_point)
                point_labels.append(point_label)
                # print(f"Random point in component {label}: {random_point}, label: {point_labels}")
        if(len(point)==1):
            point.append(point[0])
            point_labels.append(point_labels[0])
        if(len(point)>2):
            point = point[:2]
            point_labels = point_labels[:2]
        point = np.array(point)
        point_labels = np.array(point_labels)
        return point_labels,point
    
    def __getitem__(self, index):
        point_label = 1

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'images', name)
        msk_path = os.path.join(self.data_path, self.mode, 'masks', name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)

        if self.prompt == 'click':
            # two prompt
            point_label, pt = self.find_connected_components(np.array(mask))
            # one prompt
            # point_label, pt = random_click(np.array(mask) / 255, point_label)

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)


            if self.transform_msk:
                mask = self.transform_msk(mask)
                
            # if (inout == 0 and point_label == 1) or (inout == 1 and point_label == 0):
            #     mask = 1 - mask
        mask = torch.clamp(mask,min=0,max=1).int()

        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }