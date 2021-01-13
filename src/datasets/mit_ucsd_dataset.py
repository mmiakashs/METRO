import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import pandas as pd
import numpy as np

import sys, os
import csv, string, re, math
from collections import defaultdict
from pathlib import Path

from src.config import config
from src.utils.noises import *
from src.utils.log import *

# features, num_features
# seq_max_len=2000
# window_size=10, window_stride=None,
# motion_type


class MIT_UCSD_Dataset(Dataset):

    def __init__(self, 
                hparams, 
                dataset_type='train',
                restricted_ids=None, restricted_labels=None,
                allowed_ids=None, allowed_labels=None,
                noisy_sampler=None):
        self.hparams = hparams
        self.dataset_type = dataset_type
        self.base_dir = self.hparams.data_file_dir_base_path
        self.motion_type = self.hparams.motion_type
        self.dataset_filename = self.hparams.dataset_filename
        self.modality_prop = self.hparams.modality_prop
        self.restricted_ids = restricted_ids
        self.restricted_labels = restricted_labels
        self.allowed_ids = allowed_ids
        self.allowed_labels = allowed_labels
        self.modalities = self.hparams.modalities
        self.noisy_sampler = noisy_sampler

        self.features = self.hparams.mit_ucsd_modality_features
        self.num_features = self.modality_prop['num_features']
        self.load_data()
        self.activity_type_names = self.data.activity_type.unique()
        self.num_activity_types, self.activity_type_id, self.activity_id_type = self.get_activity_type_id(self.activity_type_names)
        
    def load_data(self):
        self.data = pd.read_csv(self.base_dir+'/'+self.dataset_filename+'.csv')
        if self.motion_type is not None:
            self.data = self.data[self.data['motion_type']==self.motion_type]

        if (self.restricted_ids != None):
            self.data = self.data[~self.data[self.restricted_labels].isin(self.restricted_ids)]

        if (self.allowed_ids != None):
            self.data = self.data[self.data[self.allowed_labels].isin(self.allowed_ids)]

        self.data.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.data)

    def pad_trim_hnd_feature(self, seq, seq_max_len):
        seq = seq[: seq_max_len, :]
        seq_len = seq.shape[0]
        feature_len = seq.shape[1]
        seq = np.concatenate((seq, np.zeros((seq_max_len - seq_len, feature_len))), axis=0)
        return seq

    def split_seq(self, seq, window_size, window_stride):
        return seq.unfold(dimension=0, size=window_size, step=window_stride)

    def __getitem__(self, idx):
        data = {}
        skip_frame_len_dict = {}
        for modality in self.modalities:
            skip_frame_len_dict[modality] = int(self.modality_prop[modality]['skip_frame_len'])
        data = self.prepare_data_element(idx, skip_frame_len_dict)
        return data

    def prepare_data_element(self, idx, skip_frame_len_dict=None):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        activity_types = []

        is_noisy = False 
        noisy_modality = None
        noisy_modality_idx = len(self.modalities)
        if self.noisy_sampler is not None:
            noisy_sample = self.noisy_sampler.sample()
            if(noisy_sample[0]==1):
                is_noisy = True
                noisy_modality_idx = random.randint(0, len(self.modalities)-1)
                noisy_modality = self.modalities[noisy_modality_idx]

        seq_len_dict = {}
        for modality in self.modalities:
            seq_len_dict[modality] = 0

        max_modality_seq_len = 0
        feature_seqs = {}
        mod_seq_max_len = 0
        for feature in self.features:
            modality = self.get_modality(feature)
            feature_attributes = self.get_feature_attributes(feature)
            seq_max_len = self.modality_prop[modality]['seq_max_len']

            tm_seq_len = 0
            if(pd.isna(self.data.loc[idx, feature])):
                temp_seq = np.zeros((seq_max_len, len(feature_attributes)))
                # print(f'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ {feature}, {modality}: no data')
            else:
                features_filename = os.path.join(self.base_dir, self.data.loc[idx, feature])
                temp_seq = np.zeros((seq_max_len, len(feature_attributes)))

                if(os.path.exists(features_filename)):
                    temp_seq_df = pd.read_csv(features_filename)
                    temp_seq_df = temp_seq_df[feature_attributes]
                    temp_seq = np.array(temp_seq_df)
                    tm_seq_len = temp_seq.shape[0]

            temp_seq = temp_seq.astype('float').reshape(-1, len(feature_attributes))
            feature_seqs[feature] = temp_seq

            seq_len_dict[modality] = max(seq_len_dict[modality], tm_seq_len)
            mod_seq_max_len = max(mod_seq_max_len, tm_seq_len)

        seqs = defaultdict(list)
        for feature in self.features:
            modality = self.get_modality(feature)
            tm_max_seq_len = seq_len_dict[modality]
            if(tm_max_seq_len==0):
                tm_max_seq_len = mod_seq_max_len
            temp_seq = self.pad_trim_hnd_feature(feature_seqs[feature],
                                                 tm_max_seq_len)
            temp_seq = torch.from_numpy(temp_seq).float()
            # print('##### before', self.data.loc[idx,['activity_type']], self.data.loc[idx,['motion_type']], modality, temp_seq.size())

            motion_type = self.data.loc[idx,['motion_type']][0]
            temp_seq = self.split_seq(temp_seq,
                                window_size=self.modality_prop[motion_type]['window_size'],
                                window_stride=self.modality_prop[motion_type]['window_stride'])

            # print('##### after', self.data.loc[idx,['activity_type']], self.data.loc[idx,['motion_type']], modality, temp_seq.size())

            if skip_frame_len_dict[modality] is not None:
                start_idx = 0
                seq_len = temp_seq.size(0)
                if self.modality_prop[modality]['is_rand_starting']:
                    start_idx = random.randint(0,max(0, seq_len-self.modality_prop[modality]['seq_max_len']))
                temp_seq = temp_seq[start_idx::skip_frame_len_dict[modality], :]

            seqs[modality].append(temp_seq)

        activity_types.append(self.data.loc[idx,['activity_type']])

        seq_data = {}
        modality_mask = []
        for modality in self.modalities:
            tm_seq = torch.stack(seqs[modality], dim=1).contiguous().float()

            if (modality==noisy_modality) and is_noisy and (self.hparams.noise_type is not None):
                if(self.hparams.noise_type=='random'):
                    addNoise = RandomNoise(noise_level=self.hparams.noise_level)
                elif(self.hparams.noise_type=='gaussian'):
                    addNoise = GaussianNoise(noise_level=self.hparams.noise_level)
                tm_seq = addNoise(tm_seq)

            seq_data[modality] = self.normalize_data(tm_seq, modality)

            if(seq_len_dict[modality]==0):
                modality_mask.append(True)
            else:
                modality_mask.append(False)

        activity_types_ids = [ config.mit_ucsd_activity_id[str(temp[0])] for temp in activity_types]
        task_types = [config.mit_ucsd_activity_task[str(temp[0])] for temp in activity_types]
        task_types_ids = [config.mit_ucsd_task_id[temp] for temp in task_types]
        modality_mask = torch.from_numpy(np.array(modality_mask)).bool()

        data = dict()
        for modality in self.modalities:
            data[modality] = seq_data[modality]
            data[modality + config.modality_seq_len_tag] = seq_data[modality].size(0)

        data['label'] = activity_types_ids[0]
        data['task_label'] = task_types_ids[0]
        data['modality_mask'] = modality_mask
        data['is_noisy'] = 1 if is_noisy else 0
        data['noisy_modality'] = noisy_modality_idx

        return data

    def get_feature_attributes(self,feature_name):
        if('hnd' in feature_name):
            return ['translation_x','translation_y','translation_z','rotation_x','rotation_y','rotation_z','rotation_w']
        elif('myo_emg' in feature_name):
            return ['e1','e2','e3','e4','e5','e6','e7','e8']
        elif('myo_imu' in feature_name):
            return ['ori_x','ori_y','ori_z','ori_w','angular_vel_x','angular_vel_y','angular_vel_z','lin_acc_x','lin_acc_y','lin_acc_z']
        elif('pose' in feature_name):
            return ['pos_x','pos_y','pos_z','ori_x','ori_y','ori_z','ori_w']

    def get_activity_type_id(self, activity_types):
        activity_types = sorted(activity_types)
        num_activity_types = len(activity_types)+1

        temp_dict_type_id = { activity_types[i]:i+1 for i in range(len(activity_types)) }
        temp_dict_id_type = { i+1 : activity_types[i] for i in range(len(activity_types)) }
        return num_activity_types, temp_dict_type_id, temp_dict_id_type

    def get_modality(self, feature_name):
        for modality in self.modalities:
            if(modality in feature_name):
                return modality
        return None

    def normalize_data(self, tm_tensor, modality):
        tm_tensor = F.normalize(tm_tensor, p=2, dim=1)
        return tm_tensor

def get_ids_from_split_utd_mhad(split_ids, split_index):
    person_ids = []
    for id in split_index:
        person_ids.append(split_ids[id])
    return person_ids

def gen_mask(seq_len, max_len):
   return torch.arange(max_len) > seq_len

class MIT_UCSD_Collator:

    def __init__(self, modalities):
        self.modalities = modalities

    def __call__(self, batch):
        batch_size = len(batch)
        data = {}
        
        for modality in self.modalities:
            data[modality] = pad_sequence([batch[bin][modality] for bin in range(batch_size)], batch_first=True)
            data[modality + config.modality_seq_len_tag] = torch.tensor(
                    [batch[bin][modality + config.modality_seq_len_tag] for bin in range(batch_size)],
                    dtype=torch.float)
            
            seq_max_len = data[modality + config.modality_seq_len_tag].max()
            seq_mask = torch.stack(
            [gen_mask(seq_len, seq_max_len) for seq_len in data[modality + config.modality_seq_len_tag]],
                dim=0)
            data[modality + config.modality_mask_suffix_tag] = seq_mask

        data['label'] = torch.tensor([batch[bin]['label'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['task_label'] = torch.tensor([batch[bin]['task_label'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['is_noisy'] = torch.tensor([batch[bin]['is_noisy'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['noisy_modality'] = torch.tensor([batch[bin]['noisy_modality'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['modality_mask'] = torch.stack([batch[bin]['modality_mask'] for bin in range(batch_size)], dim=0).bool()
        return data