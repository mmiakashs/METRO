import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import config
from collections import defaultdict
from src.models.KeylessAttention import KeylessAttention

class TaskRouter(nn.Module):
    def __init__(self, hparams):

        super(TaskRouter, self).__init__()
        self.hparams = hparams
        self.modality_embedding_size = self.hparams.indi_modality_embedding_size
        self.task_list = self.hparams.task_list
        self.modalities = self.hparams.modalities
        self.num_modality = len(self.modalities)
        self.multi_modal_nhead = self.hparams.multi_modal_nhead
        self.dropout = self.hparams.task_head_mm_fusion_dropout

        if self.hparams.dataset_name=='uva_dar':
            self.task_activity_map = None
        elif self.hparams.dataset_name=='mit_ucsd':
            self.task_activity_map = config.mit_ucsd_task_activity

        self.modality_router = nn.ModuleDict()
        for modality in self.modalities:
            for task in self.task_list:
                router_key = f'{task}_{modality}'
                self.modality_router[router_key] = nn.Linear(self.modality_embedding_size, 1)

        self.mm_fuser = nn.ModuleDict()
        for task in self.task_list:
            if self.hparams.mm_fusion_attention_type=='multi_head':
                self.mm_fuser[task] = nn.MultiheadAttention(embed_dim=self.modality_embedding_size,
                                                    num_heads=self.multi_modal_nhead,
                                                    dropout=self.dropout)
            elif self.hparams.mm_fusion_attention_type=='keyless':
                self.mm_fuser[task] = KeylessAttention(self.modality_embedding_size)
            
            self.mm_mhattn_relu = nn.ReLU()
            self.mm_mhattn_dropout = nn.Dropout(p=self.dropout)

        self.har_classifier = nn.ModuleDict()
        for task in self.task_list:
            self.har_classifier[task] = nn.Linear(self.modality_embedding_size, 
                                        self.hparams.total_activities)
            self.har_classifier[task].apply(self.init_weights)
    
    def forward(self, module_out, task): 
        
        task_mm_embed_list = []
        for modality in self.modalities:
            router_key = f'{task}_{modality}'
            logits = self.modality_router[router_key](module_out[modality])
            is_modality = F.gumbel_softmax(logits, tau=1, hard=True)
            
            if is_modality[0][0] > 0:
                task_mm_embed_list.append(module_out[modality])
        
        mm_embeddings = torch.stack(task_mm_embed_list, dim=1).contiguous()
        mm_embeddings = F.relu(mm_embeddings)
        nbatches = mm_embeddings.shape[0]

        if self.num_modality>1 and self.hparams.mm_fusion_attention_type is not None:
            if self.hparams.mm_fusion_attention_type=='multi_head':
                # transpose batch and sequence (B x S x ..) --> (S x B x ..)
                mm_embeddings = mm_embeddings.transpose(0, 1).contiguous()

                # remove the modality mask batch['modality_mask']
                # so that the model can learn by itself that 
                # how it can learn in the presence of missing modality
                mm_embeddings, self.mm_attn_weight = self.mm_fuser[task](mm_embeddings, 
                                                                mm_embeddings, 
                                                                mm_embeddings)
                # transpose batch and sequence (S x B x ..) --> (B x S x ..)
                mm_embeddings = mm_embeddings.transpose(0, 1).contiguous()  
                
            elif self.hparams.mm_fusion_attention_type=='keyless':
                mm_embeddings, self.mm_attn_weight = self.mm_fuser[task](mm_embeddings)

            mm_embeddings = self.mm_mhattn_dropout(self.mm_mhattn_relu(mm_embeddings))
        
        mm_embeddings = torch.sum(mm_embeddings, dim=1).squeeze(dim=1)
        mm_embeddings = mm_embeddings.contiguous().view(nbatches, -1)
        mm_embed = F.relu(mm_embeddings)

        logits = self.har_classifier[task](mm_embeddings)
        return logits

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
