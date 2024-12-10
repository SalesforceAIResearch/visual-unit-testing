"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import copy
import os
import random

from torch.utils.data import Dataset

from my_datasets.utils import load_annotation_file

class BaseDataset(Dataset):
    def __init__(self, 
                 annotation_file,
                 image_root,
                 num_samples=-1,
                 sample_ids_file=None, 
                 key='sample_id'):
        
        self.data = load_annotation_file(annotation_file)
        if num_samples > 0:
            self.data = random.sample(self.data, num_samples)
        self.image_root = image_root 
        
        if key != 'sample_id':
            for d in self.data:
                d['sample_id'] = d[key]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ann = copy.deepcopy(self.data[index])
        return ann
