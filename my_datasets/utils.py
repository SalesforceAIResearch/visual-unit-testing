"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
import pandas as pd
import gzip


def load_annotation_file(annotation_file):
    if 'json' in annotation_file and 'jsonl' not in annotation_file:
        if '.gz' in annotation_file:
            with gzip.open(annotation_file, 'rb') as f:
                data = json.load(f)
        else:
            data = json.load(open(annotation_file))
    elif 'jsonl' in annotation_file:
        if '.gz' in annotation_file:
            with gzip.open(annotation_file, 'rb') as f:
                data = [json.loads(line.strip()) for line in f.readlines()]
        else:
            data = [json.loads(line.strip()) for line in open(annotation_file).readlines()]
    elif 'csv' in annotation_file:
        data = pd.read_csv(annotation_file)
    elif 'tsv' in annotation_file:
        data = pd.read_csv(annotation_file, delimiter='\t')
    if isinstance(data, dict):
        data = [[{"sample_id": k, **v}] if isinstance(v, dict)
                else [{"sample_id": k, **v_} for v_ in v] for k, v in data.items()]
        # flatten
        data = [d for d_ in data for d in d_]
    elif isinstance(data, pd.DataFrame):
        data = data.to_dict('records')  
    return data

def filter_by_sample_ids(data, sample_ids_file, key='questionId'):
    sample_ids = [line.strip() for line in open(sample_ids_file).readlines()]
    data = [d for d in data if str(d[key]) in sample_ids]
    return data