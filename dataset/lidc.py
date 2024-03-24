import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from utils import random_box, random_click


class LIDC(Dataset):
    names = []
    images = []
    labels = []
    series_uid = []

    def __init__(self, data_path, transform=None, transform_msk = None, prompt = 'click'):
        self.prompt = prompt
        self.transform = transform
        self.transform_msk = transform_msk
        
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(data_path):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                file_path = data_path + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
                
        
        for key, value in data.items():
            self.names.append(key)
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels) == len(self.series_uid))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        point_label = 1

        """Get the images"""
        img = np.expand_dims(self.images[index], axis=0)
        name = self.names[index]
        multi_rater = self.labels[index]

        # first click is the target most agreement among raters, otherwise, background agreement
        if self.prompt == 'click':
            point_label, pt = random_click(np.array(np.mean(np.stack(multi_rater), axis=0)) / 255, point_label)

        # Convert image (ensure three channels) and multi-rater labels to torch tensors
        img = torch.from_numpy(img).type(torch.float32)
        img = img.repeat(3, 1, 1) 
        multi_rater = [torch.from_numpy(single_rater).type(torch.float32) for single_rater in multi_rater]

        multi_rater = torch.stack(multi_rater, dim=0)
        multi_rater = multi_rater.unsqueeze(1)

        if self.prompt == 'box':
            x_min, x_max, y_min, y_max = random_box(multi_rater)
            box = [x_min, x_max, y_min, y_max]

        mask = multi_rater.mean(dim=0) # average

        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'multi_rater': multi_rater,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'box': box,
            'image_meta_dict':image_meta_dict,
        }

