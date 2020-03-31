import os

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.utils import config


#imageio read frame as (channel, height, width)
class Video:
    def __init__(self, path, seq_max_len=None, transforms=None,
                 segment_stride=130, consecutive_frame_stride=3,
                 segment_size=3):
        self.path = path
        self.seq_max_len = seq_max_len
        self.transforms = transforms
        self.container = imageio.get_reader(path, 'ffmpeg')
        self.length = self.container.count_frames()
        self.fps = self.container.get_meta_data()['fps']
        self.segment_stride = segment_stride
        self.consecutive_frame_stride = consecutive_frame_stride
        self.segment_size = segment_size

    def get_all_frames(self):
        self.init_head()
        frames = []
        face_frames = []

        tm_frame_count = 0
        total_frames_in_segment = self.consecutive_frame_stride * self.segment_size
        segment_number = 1
        for idx, frame in enumerate(self.container):
            if(tm_frame_count < total_frames_in_segment):
                if(tm_frame_count % self.consecutive_frame_stride==0):
                    frame = Image.fromarray(frame)
                    if (self.transforms != None):
                        frame = self.transforms(frame)
                    frames.append(frame)
                
            tm_frame_count += 1
            if (idx >= segment_number * self.segment_stride):
                tm_frame_count = 0
                segment_number += 1


        seq = torch.stack(frames, dim=0).float()
        seq_len = seq.size(0)
        self.container.close()
        return seq, seq_len
    
    def init_head(self):
        self.container.set_image_index(0)

    def next_frame(self):
        self.container.get_next_data()

    def get(self, key):
        return self.container.get_data(key)

    def __call__(self, key):
        return self.get(key)

    def __len__(self):
        return self.length


class UVA_DAR_Dataset(Dataset):

    def __init__(self, data_dir_base_path,
                 embed_dir_base_path,
                 modalities,
                 window_size=1, window_stride=1,
                 seq_max_len=60, transforms_modalities=None,
                 restricted_ids=None, restricted_labels=None,
                 allowed_ids=None, allowed_labels=None,
                 metadata_filename='train.csv',
                 is_pretrained_fe=False):

        self.data_dir_base_path = data_dir_base_path
        self.embed_dir_base_path = embed_dir_base_path
        self.transforms_modalities = transforms_modalities
        self.restricted_ids = restricted_ids
        self.restricted_labels = restricted_labels
        self.allowed_ids = allowed_ids
        self.allowed_labels = allowed_labels

        self.modalities = modalities
        self.seq_max_len = seq_max_len
        self.metadata_filename = metadata_filename
        self.is_pretrained_fe = is_pretrained_fe

        self.window_size = window_size
        if (window_stride == None):
            self.window_stride = window_size
        else:
            self.window_stride = window_stride

        self.load_data()
        self.activity_names = self.data.activity.unique()
        self.num_activities, self.activity_name_id, self.activity_id_name = self.get_activity_name_id(self.activity_names)

    def load_data(self):
        self.data = pd.read_csv(self.data_dir_base_path+'/'+self.metadata_filename)
        if(self.is_pretrained_fe):
            data_dir_base_path = self.embed_dir_base_path
            file_ext = '.pt'
        else:
            data_dir_base_path=self.data_dir_base_path
            file_ext = '.MP4'

        for i, row in self.data.iterrows():
            tm_filename = row[config.inside_modality_tag][:-4]
            tm_filename = f'{tm_filename}{file_ext}'
            if (not os.path.exists(f'{data_dir_base_path}/{row[config.activity_tag]}/{tm_filename}')):
                self.data.at[i, config.activity_tag] = 'MISSING'
                # print('missing inside file:',row[config.inside_modality_tag], row[config.activity_tag])

            tm_filename = row[config.outside_modality_tag][:-4]
            tm_filename = f'{tm_filename}{file_ext}'
            if ((not os.path.exists(f'{data_dir_base_path}/{row[config.activity_tag]}/{tm_filename}'))):
                self.data.at[i, config.activity_tag] = 'MISSING'
                # print('missing outside file:',row[config.outside_modality_tag], row[config.activity_tag])

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
        if (self.is_pretrained_fe):
            filename = f'{self.data.loc[idx, modality][:-4]}.pt'
            activity = self.data.loc[idx, config.activity_tag]
            data_filepath = f'{self.embed_dir_base_path}/{activity}/{filename}'
            seq = torch.load(data_filepath).detach()
            seq_len = seq.size(0)
        else:
            filename = self.data.loc[idx, modality]
            activity = self.data.loc[idx, config.activity_tag]
            data_filepath = f'{self.data_dir_base_path}/{activity}/{filename}'
            frame_parser = Video(path=data_filepath,
                                 transforms=self.transforms_modalities[modality],
                                 seq_max_len=self.seq_max_len)

            seq, seq_len = frame_parser.get_all_frames()
        if ((self.seq_max_len is not None) and (seq_len > self.seq_max_len)):
            seq = seq[:self.seq_max_len, :]
            seq_len = self.seq_max_len

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
        data[config.activity_tag] = self.activity_name_id[str(data_label)]
        data[config.modality_mask_tag] = modality_mask

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

def pad_collate(batch):
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
    
    data[config.activity_tag] = torch.tensor([batch[bin][config.activity_tag] for bin in range(batch_size)],
                                dtype=torch.long)
    data['modality_mask'] = torch.stack([batch[bin]['modality_mask'] for bin in range(batch_size)], dim=0).bool()
    return data
