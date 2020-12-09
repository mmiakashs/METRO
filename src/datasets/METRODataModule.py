import pytorch_lightning as pl
from src.datasets.uva_dar_dataset import *
from src.utils.log import *
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import math

class METRODataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        if self.hparams.dataset_name=='uva_dar':
            self.Dataset = UVA_DAR_Dataset
            self.collate_fn = UVA_DAR_Collator(self.hparams.modalities)

        self.txt_logger = TextLogger(self.hparams.log_base_dir, 
                                    self.hparams.log_filename,
                                    print_console=True)


    # def prepare_data(self):
    def setup(self, stage=None):
        
        if (self.hparams.data_split_type=='fixed_subject' or self.hparams.data_split_type=='cross_subject') and (not self.hparams.share_train_dataset):
            self.train_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='train',
                                            restricted_ids = self.hparams.train_restricted_ids,
                                            restricted_labels = self.hparams.train_restricted_labels,
                                            allowed_ids=self.hparams.train_allowed_ids,
                                            allowed_labels=self.hparams.train_allowed_labels)

            self.valid_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='valid',
                                            restricted_ids = self.hparams.valid_restricted_ids,
                                            restricted_labels = self.hparams.valid_restricted_labels,
                                            allowed_ids=self.hparams.valid_allowed_ids,
                                            allowed_labels=self.hparams.valid_allowed_labels)

            self.test_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='test',
                                            restricted_ids = self.hparams.test_restricted_ids,
                                            restricted_labels = self.hparams.test_restricted_labels,
                                            allowed_ids=self.hparams.test_allowed_ids,
                                            allowed_labels=self.hparams.test_allowed_labels)
        
        elif (self.hparams.data_split_type=='fixed_subject' or self.hparams.data_split_type=='cross_subject') and self.hparams.share_train_dataset:
            self.train_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='train',
                                            restricted_ids = self.hparams.train_restricted_ids,
                                            restricted_labels = self.hparams.train_restricted_labels,
                                            allowed_ids=self.hparams.train_allowed_ids,
                                            allowed_labels=self.hparams.train_allowed_labels)

            dataset_len = len(self.train_dataset)
            self.hparams.dataset_len = dataset_len

            valid_len = math.floor(dataset_len*self.hparams.valid_split_pct)
            train_len = dataset_len - valid_len

            self.train_dataset, self.valid_dataset = random_split(self.train_dataset,
                                                                [train_len, valid_len])

            self.test_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='test',
                                            restricted_ids = self.hparams.test_restricted_ids,
                                            restricted_labels = self.hparams.test_restricted_labels,
                                            allowed_ids=self.hparams.test_allowed_ids,
                                            allowed_labels=self.hparams.test_allowed_labels)
        
        elif self.hparams.data_split_type=='session':
            full_dataset = self.Dataset(hparams=self.hparams, 
                                        noisy_sampler=train_noisy_sampler)
            dataset_len = len(full_dataset)
            self.hparams.dataset_len = dataset_len

            test_len = math.floor(dataset_len*self.hparams.test_split_pct)
            valid_len = math.floor((dataset_len-test_len)*self.hparams.valid_split_pct)
            train_len = dataset_len - valid_len - test_len

            self.train_dataset, self.valid_dataset, self.test_dataset = random_split(full_dataset,
                                                                            [train_len, valid_len, test_len])

        # self.txt_logger.log(f'train subject ids: {sorted(self.train_dataset.data[self.hparams.train_element_tag].unique())}\n')
        # self.txt_logger.log(f'valid subject ids: {sorted(self.valid_dataset.data[self.hparams.train_element_tag].unique())}\n')
        # self.txt_logger.log(f'test subject ids: {sorted(self.test_dataset.data[self.hparams.train_element_tag].unique())}\n')
        self.txt_logger.log(f'train dataset len: {len(self.train_dataset)}\n')
        self.txt_logger.log(f'valid dataset len: {len(self.valid_dataset)}\n')
        self.txt_logger.log(f'test dataset len: {len(self.test_dataset)}\n')

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,
                            batch_size=self.hparams.batch_size,
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            drop_last=True)
        return loader

    def val_dataloader(self):
        if self.hparams.no_validation:
            return None
            
        loader = DataLoader(self.valid_dataset,
                            batch_size=min(self.hparams.batch_size, 2),
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            drop_last=True)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset,
                            batch_size=min(self.hparams.batch_size, 2),
                            collate_fn=self.collate_fn,
                            num_workers=self.hparams.num_workers,
                            drop_last=True)
        return loader