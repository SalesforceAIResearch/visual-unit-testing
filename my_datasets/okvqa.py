"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from collections import defaultdict
import random
import pickle
import os
from tqdm import tqdm

from my_datasets.base import BaseDataset
from my_datasets.utils import filter_by_sample_ids


class OKVQADataset(BaseDataset):
    def __init__(self, 
                 annotation_file, 
                 image_root, 
                 num_samples=-1, 
                 sample_ids_file=None, 
                 key='question_id',
                 sample_per_type=-1):
        super().__init__(annotation_file, image_root, num_samples, sample_ids_file, key)
        if sample_ids_file:
            self.data = filter_by_sample_ids(self.data, sample_ids_file, key)
        if sample_per_type is not None and sample_per_type>0:
            split = 'train' if 'train' in annotation_file else 'val' if 'val' in annotation_file else 'test'
            samples_dir = './my_datasets/samples/OKVQA'
            os.makedirs(samples_dir, exist_ok=True)
            sample_per_type_file = os.path.join(samples_dir, f'{split}_{sample_per_type}.p')
            if os.path.exists(sample_per_type_file):
                self.data = pickle.load(open(sample_per_type_file, 'rb'))
            else:
                print(f'Sampling {sample_per_type} data per type...')
                id2type = {d['question_id']: d['question_type'] for d in tqdm(self.data)}
                types = list(set(id2type.values()))
                type2id = defaultdict(list)
                for type in types:
                    type2id[type] = [i for i in id2type if id2type[i]==type]
                for k, v in type2id.items():
                    type2id[k] = set(random.sample(v, min(sample_per_type, len(v))))
                self.data = [d for d in self.data \
                    if d['question_id'] in type2id[d['question_type']]]
                pickle.dump(self.data, open(sample_per_type_file, 'wb'))
        
        for d in self.data:
            d['image'] = d['image_id']
            d['answer']= [d_['raw_answer'] for d_ in d['answers']]
        
    
        
    def __getitem__(self, index):
        ann = super().__getitem__(index)
        ann['image_path']  = os.path.join(self.image_root, 'COCO_val2014_' + str(ann['image_id']).zfill(12) + '.jpg')
        ann['text'] = ann['question']
        # del ann['question']
        return ann