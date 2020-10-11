import os

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.config import config
from src.utils.log import *
from src.datasets.video import Video


class UVA_DAR_Dataset(Dataset):

    def __init__(self,
                 hparams,
                 dataset_type='train',
                 restricted_ids=None, restricted_labels=None,
                 allowed_ids=None, allowed_labels=None):

        self.hparams = hparams
        self.dataset_type = dataset_type
        self.base_dir = self.hparams.data_file_dir_base_path
        self.embed_dir_base = self.hparams.embed_dir_base_path
        self.dataset_filename = self.hparams.dataset_filename
        self.modality_prop = self.hparams.modality_prop
        self.transforms_modalities = self.hparams.transforms_modalities
        self.restricted_ids = restricted_ids
        self.restricted_labels = restricted_labels
        self.allowed_ids = allowed_ids
        self.allowed_labels = allowed_labels
        self.modalities = self.hparams.modalities

        self.load_data()
        self.activity_names = self.data.activity.unique()
        self.num_activity_types, self.activity_name_id, self.activity_id_name = self.get_activity_name_id(self.activity_names)

    def load_data(self):

        self.data = pd.read_csv(self.base_dir+'/'+self.dataset_filename)
        if self.modality_prop['is_pretrained_fe']:
            base_dir = self.embed_dir_base
            file_ext = '.pt'
        else:
            base_dir=self.base_dir
            file_ext = '.MP4'

        for i, row in self.data.iterrows():
            for modality in self.modalities:
                tm_filename = row[modality]
                tm_filename = f'{tm_filename}{file_ext}'
                
                if (not os.path.exists(f'{base_dir}/{row[config.activity_tag]}/{tm_filename}')):
                    self.data.at[i, config.activity_tag] = 'MISSING'
                    # print(row[modality])
                    # print(f'{base_dir}/{row[config.activity_tag]}/{tm_filename}')
                    print(f'missing {modality} file:',row[modality], row[config.activity_tag])
        
        self.data = self.data[self.data[config.activity_tag] != 'MISSING']

        if (self.restricted_ids is not None):
            self.data = self.data[~self.data[self.restricted_labels].isin(self.restricted_ids)]

        if (self.allowed_ids is not None):
            self.data = self.data[self.data[self.allowed_labels].isin(self.allowed_ids)]

        self.data.reset_index(inplace=True)

    def __len__(self):
        return len(self.data)

    def gen_mask(self, seq_len, max_len):
        return torch.arange(max_len) > seq_len
    
    def get_video_data(self, idx, modality):
        if self.modality_prop[modality]['is_pretrained_fe']:
            filename = f'{self.data.loc[idx, modality]}.pt'
            activity = self.data.loc[idx, config.activity_tag]
            data_filepath = f'{self.embed_dir_base}/{activity}/{filename}'
            seq = torch.load(data_filepath, map_location='cpu').detach()
            seq_len = seq.size(0)
        else:
            filename = f'{self.data.loc[idx, modality]}.MP4'
            activity = self.data.loc[idx, config.activity_tag]
            data_filepath = f'{self.base_dir}/{activity}/{filename}'
            frame_parser = Video(path=data_filepath,
                                 transforms=self.transforms_modalities[modality],
                                 seq_max_len=self.modality_prop[modality]['seq_max_len'],
                                 skip_frame_ratio = self.modality_prop[modality]['skip_frame_ratio'],
                                 skip_frame_len = int(self.modality_prop[modality]['skip_frame_len']),
                                 is_rand_starting=self.modality_prop[modality]['is_rand_starting'])

            seq, seq_len = frame_parser.get_all_frames()

        return seq, seq_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_label = self.data.loc[idx, config.activity_tag]
        data = {}
        modality_mask = []
        
        # print(f'************ Start Data Loader for {idx} ************')
        for modality in self.modalities:
            seq, seq_len = self.get_video_data(idx, modality)
            data[modality] = seq
            data[modality + config.modality_seq_len_tag] = seq_len
            modality_mask.append(True if seq_len == 0 else False)

        modality_mask = torch.from_numpy(np.array(modality_mask)).bool()
        data['label'] = self.activity_name_id[str(data_label)]
        data['modality_mask'] = modality_mask

        # print(f'************ End Data Loader for {idx} ************')
        return data

    def get_activity_name_id(self, activity_names):

        activity_names = sorted(activity_names)
        num_labels = len(activity_names)

        temp_dict_type_id = {activity_names[i]: i for i in range(len(activity_names))}
        temp_dict_id_type = {i : activity_names[i] for i in range(len(activity_names))}
        return num_labels, temp_dict_type_id, temp_dict_id_type

def get_ids_from_split(split_ids, split_index):
    person_ids = []
    for id in split_index:
        person_ids.append(split_ids[id])
    return person_ids

modalities = [config.inside_modality_tag,
              config.outside_modality_tag]

def gen_mask(seq_len, max_len):
    return torch.arange(max_len) > seq_len

class UVA_DAR_Collator:

    def __init__(self, modalities):
        self.modalities = modalities

    def __call__(self, batch):
        batch_size = len(batch)
        data = {}
        for modality in modalities:
            data[modality] = pad_sequence([batch[bin][modality] for bin in range(batch_size)], batch_first=True)
            data[modality + config.modality_seq_len_tag] = torch.tensor(
                [batch[bin][modality + config.modality_seq_len_tag] for bin in range(batch_size)],
                dtype=torch.float)
            
    #         print(f'{modality} seq lengths: ',data[modality + config.modality_seq_len_tag])

            seq_max_len = data[modality + config.modality_seq_len_tag].max()
            seq_mask = torch.stack(
                [gen_mask(seq_len, seq_max_len)  for seq_len in data[modality + config.modality_seq_len_tag]], dim=0)
            data[modality + config.modality_mask_suffix_tag] = seq_mask
        
        data['label'] = torch.tensor([batch[bin]['label'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['modality_mask'] = torch.stack([batch[bin]['modality_mask'] for bin in range(batch_size)], dim=0).bool()
        return data
