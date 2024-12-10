"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from .gqa import GQADataset
from .winoground import WinoGroundDataset
from .sugarcrepe import SugarCREPEDataset
from .vqa import VQADataset
from .okvqa import OKVQADataset

def get_dataset_class(dataset_name):
    if dataset_name == 'GQA':
        return GQADataset
    elif dataset_name == 'WinoGround':
        return WinoGroundDataset
    elif dataset_name == 'SugarCREPE':
        return SugarCREPEDataset
    elif dataset_name == 'VQA':
        return VQADataset
    elif dataset_name == 'OKVQA':
        return OKVQADataset
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")