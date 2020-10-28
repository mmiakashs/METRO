import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from pytorch_lightning.overrides.data_parallel import (
    LightningDistributedDataParallel,
    LightningDataParallel,
)
from sklearn.metrics import f1_score, precision_score, recall_score
import math
from src.datasets.uva_dar_dataset import *
from src.utils.log import *
from src.models.MM_Encoder import MM_Encoder
from .HAR_Classification import HAR_Classification
from src.utils.log import TextLogger
from src.utils.model_checkpointing import ModelCheckpointing
from collections import Counter


class UVA_METRO_Model(pl.LightningModule):
    def __init__(self,
                 hparams):

        super(UVA_METRO_Model, self).__init__()

        self.hparams = hparams
        self.modality_prop = self.hparams.modality_prop
        self.modalities = self.hparams.modalities
        self.num_modality = len(self.modalities)
        self.mm_embedding_attn_merge_type = self.hparams.mm_embedding_attn_merge_type
        self.lstm_bidirectional = self.hparams.lstm_bidirectional
        self.modality_embedding_size = self.hparams.lstm_hidden_size
        self.num_activity_types = self.hparams.num_activity_types
        self.weights_loss = []

        if self.hparams.dataset_name=='uva_dar':
            self.Dataset = UVA_DAR_Dataset
            self.collate_fn = UVA_DAR_Collator(self.hparams.modalities)

        # build sub-module of the learning model
        if self.hparams.modality_encoder_type==config.mm_attn_encoder:
            self.mm_encoder = MM_Encoder(self.hparams)

        self.har_classification = HAR_Classification(self.hparams.indi_modality_embedding_size, 
                                            self.num_activity_types)
        
        # define the metrics and the checkpointing mode
        metrics_mode_dict = {'loss': 'min',
                            'accuracy': 'max',
                            'f1_scores': 'max',
                            'precision': 'max',
                            'recall_scores': 'max'}
        train_metrics_save_ckpt_mode = {'train_loss': True}
        valid_metrics_save_ckpt_mode = {'valid_loss': True,
                                    'valid_accuracy': True,
                                    'valid_f1_scores': True}
        train_metrics_mode_dict = {}
        valid_metrics_mode_dict = {}
        train_metrics = []
        valid_metrics = []

        for metric in metrics_mode_dict:
            train_metrics.append(f'train_{metric}')
            valid_metrics.append(f'valid_{metric}')
            train_metrics_mode_dict[f'train_{metric}'] = metrics_mode_dict[metric]
            valid_metrics_mode_dict[f'valid_{metric}'] = metrics_mode_dict[metric]
        
        self.txt_logger = TextLogger(self.hparams.log_base_dir, 
                                    self.hparams.log_filename,
                                    print_console=True)
        self.train_model_checkpointing = ModelCheckpointing(self.hparams.model_save_base_dir,
                                                self.hparams.model_checkpoint_filename,
                                                train_metrics,
                                                train_metrics_save_ckpt_mode,
                                                train_metrics_mode_dict,
                                                self.txt_logger)
        
        self.valid_model_checkpointing = ModelCheckpointing(self.hparams.model_save_base_dir,
                                                self.hparams.model_checkpoint_filename,
                                                valid_metrics,
                                                valid_metrics_save_ckpt_mode,
                                                valid_metrics_mode_dict,
                                                self.txt_logger)

        self.test_log = None
        self.mm_embed = None
    
    def forward(self, batch):
        module_out, mm_embed = self.mm_encoder(batch)
        self.mm_embed = mm_embed
        return self.har_classification(mm_embed)

    def set_parameter_requires_grad(self, model, is_require):
        for param in model.parameters():
            param.requires_grad = is_require

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        labels = batch['label']
        batch_size = batch['label'].size(0)
            
        output = self(batch)

        _, preds = torch.max(F.softmax(output, dim=1), 1)
        accuracy = torch.sum(preds == labels.data) / float(batch_size)
        loss = self.loss_fn(output, labels)

        tm_preds = preds.cpu().data.numpy()
        tm_labels = labels.cpu().data.numpy()

        f1_scores = torch.tensor(f1_score(tm_preds, tm_labels, average='micro'))
        precision = torch.tensor(precision_score(tm_preds, tm_labels, average='micro'))
        recall_scores = torch.tensor(recall_score(tm_preds, tm_labels, average='micro'))

        return {'loss': loss,
                'log':{
                    'loss': loss,
                    'accuracy': accuracy,
                    'f1_scores': f1_scores,
                    'precision': precision,
                    'recall_scores': recall_scores}
                }

    def training_epoch_end(self, outputs):
        full_results = self.gen_metric_epoch_end(outputs, pre_log_tag='train')
        results = full_results['log']
        is_dp_module = isinstance(self, (LightningDistributedDataParallel,
                                         LightningDataParallel))
        model = self.module if is_dp_module else self
        print(results)
        self.train_model_checkpointing.update_metric_save_ckpt(results, model)
        return full_results

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        full_results = self.gen_metric_epoch_end(outputs, pre_log_tag='valid')
        results = full_results['log']
        is_dp_module = isinstance(self, (LightningDistributedDataParallel,
                                         LightningDataParallel))
        model = self.module if is_dp_module else self
        self.valid_model_checkpointing.update_metric_save_ckpt(results, model)
        return full_results

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        results = self.gen_metric_epoch_end(outputs, pre_log_tag='test')
        self.test_log = results['log']
        self.txt_logger.log(f'{str(results)}\n')
        return results

    def eval_step(self, batch, batch_idx):

        labels = batch['label']
        batch_size = batch['label'].size(0)

        output = self(batch)

        _, preds = torch.max(F.softmax(output, dim=1), 1)
        accuracy = torch.sum(preds == labels.data) / float(batch_size)
        loss = self.loss_fn(output, labels)

        tm_preds = preds.cpu().data.numpy()
        tm_labels = labels.cpu().data.numpy()

        f1_scores = torch.tensor(f1_score(tm_preds, tm_labels, average='micro'))
        precision = torch.tensor(precision_score(tm_preds, tm_labels, average='micro'))
        recall_scores = torch.tensor(recall_score(tm_preds, tm_labels, average='micro'))
        full_results = {'loss': loss,
                        'accuracy': accuracy,
                        'f1_scores': f1_scores,
                        'precision': precision,
                        'recall_scores': recall_scores}
        return {'log': full_results}

    def gen_metric_epoch_end(self, outputs, pre_log_tag=""):
        loss = []
        accuracy = []
        f1_scores = []
        precision = []
        recall_scores = []

        for output in outputs:
            if 'log' in output:
                output = output['log']
            else:
                output = output['log_metrics']

            loss.append(output['loss'])
            if 'accuracy' in output:
                accuracy.append(output['accuracy'])
            if 'f1_scores' in output:
                f1_scores.append(output['f1_scores'])
            if 'precision' in output:
                precision.append(output['precision'])
            if 'recall_scores' in output:
                recall_scores.append(output['recall_scores'])
        loss = torch.mean(torch.tensor(loss))
        accuracy = torch.mean(torch.tensor(accuracy))
        f1_scores = torch.mean(torch.tensor(f1_scores))
        precision = torch.mean(torch.tensor(precision))
        recall_scores = torch.mean(torch.tensor(recall_scores))

        results = {'log': {
            f'epoch': self.current_epoch,
            f'{pre_log_tag}_loss': loss.item(),
            f'{pre_log_tag}_accuracy': accuracy.item(),
            f'{pre_log_tag}_f1_scores': f1_scores.item(),
            f'{pre_log_tag}_precision': precision.item(),
            f'{pre_log_tag}_recall_scores': recall_scores.item()}}
        return results

    def configure_optimizers(self):
        self.loss_fn = nn.CrossEntropyLoss()#(weight=torch.FloatTensor(self.weights_loss))
        # model_params = list(self.mm_encoder.parameters()) + list(self.har_classification.parameters())
        model_params = self.parameters()
        optimizer = torch.optim.AdamW(model_params, lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                        T_0=self.hparams.cycle_length,
                                                                        T_mult=self.hparams.cycle_mul)
        return [optimizer], [lr_scheduler]
            
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

    def prepare_data(self):
        
        if (self.hparams.data_split_type=='cross_subject' or self.hparams.data_split_type=='fixed_subject') and (not self.hparams.share_train_dataset):
            self.train_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='train',
                                            restricted_ids = self.hparams.train_restricted_ids,
                                            restricted_labels = self.hparams.train_restricted_labels,
                                            allowed_ids=self.hparams.train_allowed_ids,
                                            allowed_labels=self.hparams.train_allowed_labels)
            dictionary_count = Counter(self.train_dataset.data["labeled"])
            for i in range(len(dictionary_count)):
                self.weights_loss.append(sum(dictionary_count.values())/dictionary_count[i])
            #print(self.train_dataset.data.columns)

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
        
        elif (self.hparams.data_split_type=='cross_subject' or self.hparams.data_split_type=='fixed_subject') and self.hparams.share_train_dataset:
            self.train_dataset = self.Dataset(hparams=self.hparams,
                                            dataset_type='train',
                                            restricted_ids = self.hparams.train_restricted_ids,
                                            restricted_labels = self.hparams.train_restricted_labels,
                                            allowed_ids=self.hparams.train_allowed_ids,
                                            allowed_labels=self.hparams.train_allowed_labels)
            dictionary_count = Counter(self.train_dataset.data["labeled"])
            for i in range(len(dictionary_count)):
                self.weights_loss.append(sum(dictionary_count.values())/dictionary_count[i])
            #print(self.train_dataset.data.columns)

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
        
        # self.txt_logger.log(f'train subject ids: {sorted(self.train_dataset.data[self.hparams.train_element_tag].unique())}\n')
        # self.txt_logger.log(f'valid subject ids: {sorted(self.valid_dataset.data[self.hparams.train_element_tag].unique())}\n')
        # self.txt_logger.log(f'test subject ids: {sorted(self.test_dataset.data[self.hparams.train_element_tag].unique())}\n')
        self.txt_logger.log(f'train dataset len: {len(self.train_dataset)}\n')
        self.txt_logger.log(f'valid dataset len: {len(self.valid_dataset)}\n')
        self.txt_logger.log(f'test dataset len: {len(self.test_dataset)}\n')
