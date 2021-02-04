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
            data_dir_base_path = self.embed_dir_base
            data_dir_base_path_csv = self.base_dir
            file_ext = '.pt'
            file_ext_csv = ".csv"
        else:
            data_dir_base_path=self.base_dir
            file_ext = '.MP4'
            file_ext_csv = ".csv"

        for i, row in self.data.iterrows():
            tm_filename = row[config.inside_modality_tag]
            tm_filename = f'{tm_filename}{file_ext}'
            if (not os.path.exists(f'{data_dir_base_path}/{row[config.activity_tag]}/{tm_filename}')):
                self.data.at[i, config.activity_tag] = 'MISSING'
                #print('missing inside file:',row[config.inside_modality_tag], row[config.activity_tag])

            tm_filename = row[config.outside_modality_tag]
            tm_filename = f'{tm_filename}{file_ext}'
            if ((not os.path.exists(f'{data_dir_base_path}/{row[config.activity_tag]}/{tm_filename}'))):
                self.data.at[i, config.activity_tag] = 'MISSING'
                #print(f'{data_dir_base_path}/{row[config.activity_tag]}/{tm_filename}')
                #print('missing outside file:',row[config.outside_modality_tag], row[config.activity_tag])
            
            tm_filename = row[config.gaze_modality_tag]
            tm_filename = f'{tm_filename}{file_ext_csv}'
            if ((not os.path.exists(f'{data_dir_base_path_csv}/{row[config.activity_tag]}/{tm_filename}'))):
                self.data.at[i, config.activity_tag] = 'MISSING'
                #print('missing gaze file:',row[config.gaze_modality_tag], row[config.activity_tag])

            tm_filename = row[config.pose_modality_tag]
            tm_filename = f'{tm_filename}{file_ext_csv}'
            if ((not os.path.exists(f'{data_dir_base_path_csv}/{row[config.activity_tag]}/{tm_filename}'))):
                #print(f'{data_dir_base_path}/{row[config.activity_tag]}/{tm_filename}')
                self.data.at[i, config.activity_tag] = 'MISSING'
                #print('missing pose file:',row[config.pose_modality_tag], row[config.activity_tag])

        self.data = self.data[self.data[config.activity_tag] != 'MISSING']
        if (self.restricted_ids is not None):
            self.data = self.data[~self.data[self.restricted_labels].isin(self.restricted_ids)]

        if (self.allowed_ids is not None):
            self.data = self.data[self.data[self.allowed_labels].isin(self.allowed_ids)]

        
        activity_names = self.data["activity"].unique()
        activity_names = sorted(activity_names)
        num_labels = len(activity_names)

        temp_dict_type_id = {activity_names[i]: i for i in range(len(activity_names))}
        
        self.data["labeled"] = self.data["activity"].map(temp_dict_type_id)
        self.data.reset_index(inplace=True)
#         if self.modality_prop['is_pretrained_fe']:
#             base_dir = self.embed_dir_base
#             file_ext = '.pt'
#         else:
#             base_dir=self.base_dir
#             file_ext = '.MP4'

#         for i, row in self.data.iterrows():
#             for modality in self.modalities:
#                 tm_filename = row[modality]
#                 tm_filename = f'{tm_filename}{file_ext}'
                
#                 if (not os.path.exists(f'{base_dir}/{row[config.activity_tag]}/{tm_filename}')):
#                     self.data.at[i, config.activity_tag] = 'MISSING'
#                     # print(row[modality])
#                     # print(f'{base_dir}/{row[config.activity_tag]}/{tm_filename}')
#                     print(f'missing {modality} file:',row[modality], row[config.activity_tag])
        
#         self.data = self.data[self.data[config.activity_tag] != 'MISSING']

#         if (self.restricted_ids is not None):
#             self.data = self.data[~self.data[self.restricted_labels].isin(self.restricted_ids)]

#         if (self.allowed_ids is not None):
#             self.data = self.data[self.data[self.allowed_labels].isin(self.allowed_ids)]

#         self.data.reset_index(inplace=True)

    def __len__(self):
        return len(self.data)

    def gen_mask(self, seq_len, max_len):
        return torch.arange(max_len) > seq_len
    
    def split_seq(self, seq, window_size, window_stride):
        return seq.unfold(dimension=0, size=window_size, step=window_stride)
    
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

    def get_gaze_data(self,idx,modality):
        file_ext_csv = ".csv"
        filename = f'{self.data.loc[idx, modality]}{file_ext_csv}'
        activity = self.data.loc[idx, config.activity_tag]
        data_filepath = f'{self.base_dir}/{activity}/{filename}'
        gaze_file = pd.read_csv(data_filepath,usecols=[' confidence', " gaze_0_x"," gaze_0_y", " gaze_0_z"," gaze_angle_x"," gaze_angle_y"," p_rx"," p_ry"," p_rz"])
        seq = gaze_file.to_numpy()
        seq_len = seq.shape[0]
        if (self.modality_prop[modality]['seq_max_len'] is not None) and (seq_len > self.modality_prop[modality]['seq_max_len']):
            seq = seq[:self.modality_prop[modality]['seq_max_len'], :]
        if (self.modality_prop[modality]['seq_max_len'] is not None) and (seq_len < self.modality_prop[modality]['seq_max_len']):
            temp = np.zeros((self.modality_prop[modality]['seq_max_len'],seq.shape[1]))
            temp[:seq.shape[0],:] = seq
            seq = temp.copy()
        seq = torch.from_numpy(seq)#torch.cuda.FloatTensor(seq)
        seq = self.split_seq(seq,self.modality_prop[modality]['window_size'],self.modality_prop[modality]['window_stride'])
        seq_len = seq.size(0)
        seq = seq[:,:,np.newaxis,:]
        seq = seq.type(torch.FloatTensor)
        return seq,seq_len


    def get_pose_data(self,idx,modality):
        file_ext_csv = ".csv"
        filename = f'{self.data.loc[idx, modality]}{file_ext_csv}'
        activity = self.data.loc[idx, config.activity_tag]
        data_filepath = f'{self.base_dir}/{activity}/{filename}'
        pose_file = pd.read_csv(data_filepath)
        #print(pose_file)
        columns = pose_file.columns
        columns_new = []
        #print(columns)
        for item in columns:
            try:
                if item.split("_")[1].startswith("p"):
                    columns_new.append(item)
            except:
                pass
        pose_file = pd.read_csv(data_filepath,usecols=columns_new)
        X_cols = [col for col in pose_file if col.startswith('x')]
        Y_cols = [col for col in pose_file if col.startswith('y')]
        C_cols = [col for col in pose_file if col.startswith('c')]
        X_pose = pose_file[X_cols]
        Y_pose = pose_file[Y_cols]
        C_pose = pose_file[C_cols]
        X_pose = X_pose.to_numpy()
        Y_pose = Y_pose.to_numpy()
        C_pose = C_pose.to_numpy()
        seq = np.stack((X_pose,Y_pose,C_pose),axis=-1)
        seq_len = seq.shape[0]
        if (self.modality_prop[modality]['seq_max_len'] is not None) and (seq_len > self.modality_prop[modality]['seq_max_len']):
            seq = seq[:self.modality_prop[modality]['seq_max_len'], :]
        if (self.modality_prop[modality]['seq_max_len'] is not None) and (seq_len < self.modality_prop[modality]['seq_max_len']):
            temp = np.zeros((self.modality_prop[modality]['seq_max_len'],seq.shape[1],seq.shape[2]))
            temp[:seq.shape[0],:] = seq
            seq = temp.copy()
        seq = torch.from_numpy(seq)#torch.cuda.FloatTensor(seq)
        #print(seq.shape)
        seq = self.split_seq(seq,self.modality_prop[modality]['window_size'],self.modality_prop[modality]['window_stride'])
        seq_len = seq.size(0)
        #seq = seq[:,np.newaxis,np.newaxis,:,:]
        seq = seq.type(torch.FloatTensor)
        return seq,seq_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_label = self.data.loc[idx, config.activity_tag]
        data = {}
        modality_mask = []
        
        # print(f'************ Start Data Loader for {idx} ************')
        for modality in self.modalities:
            if modality in ["inside","outside"]:
                seq, seq_len = self.get_video_data(idx, modality)
                data[modality] = seq
                data[modality + config.modality_seq_len_tag] = seq_len
                modality_mask.append(True if seq_len == 0 else False)
            if modality=="gaze":
                seq,seq_len = self.get_gaze_data(idx,modality)
                data[modality] = seq
                data[modality + config.modality_seq_len_tag] = seq_len
                modality_mask.append(True if seq_len == 0 else False)
            if modality=="pose":
                seq,seq_len = self.get_pose_data(idx,modality)
                data[modality] = seq
                data[modality + config.modality_seq_len_tag] = seq_len
                modality_mask.append(True if seq_len == 0 else False)

        modality_mask = torch.from_numpy(np.array(modality_mask)).bool()
        task_type = config.uva_metro_activity_task[str(data_label)]
        task_type_id = config.uva_metro_task_id[task_type]

        data['label'] = config.uva_metro_activity_id[str(data_label)]
        data['task_label'] = task_type_id
        data['modality_mask'] = modality_mask
        #data[config.activity_tag] = self.activity_name_id[str(data_label)]
        #data[config.modality_mask_tag] = modality_mask
#         for modality in self.modalities:
#             seq, seq_len = self.get_video_data(idx, modality)
#             data[modality] = seq
#             data[modality + config.modality_seq_len_tag] = seq_len
#             modality_mask.append(True if seq_len == 0 else False)

#         modality_mask = torch.from_numpy(np.array(modality_mask)).bool()
#         data['label'] = self.activity_name_id[str(data_label)]
#         data['modality_mask'] = modality_mask

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
              config.outside_modality_tag,
              config.gaze_modality_tag]

def gen_mask(seq_len, max_len):
    return torch.arange(max_len) > seq_len

class UVA_DAR_Collator:

    def __init__(self, modalities):
        self.modalities = modalities

    def __call__(self, batch):
        batch_size = len(batch)
        data = {}
        for modality in self.modalities:
            #print(modality)
            data[modality] = pad_sequence([batch[bin][modality] for bin in range(batch_size)], batch_first=True)
            data[modality + config.modality_seq_len_tag] = torch.tensor(
                [batch[bin][modality + config.modality_seq_len_tag] for bin in range(batch_size)],
                dtype=torch.float)
            
            #print(f'{modality} seq lengths: ',data[modality + config.modality_seq_len_tag])

            seq_max_len = data[modality + config.modality_seq_len_tag].max()
            seq_mask = torch.stack(
                [gen_mask(seq_len, seq_max_len)  for seq_len in data[modality + config.modality_seq_len_tag]], dim=0)
            data[modality + config.modality_mask_suffix_tag] = seq_mask
        
        data['label'] = torch.tensor([batch[bin]['label'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['task_label'] = torch.tensor([batch[bin]['task_label'] for bin in range(batch_size)],
                                    dtype=torch.long)
        data['modality_mask'] = torch.stack([batch[bin]['modality_mask'] for bin in range(batch_size)], dim=0).bool()
        return data
