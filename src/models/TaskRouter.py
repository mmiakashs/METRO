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
        self.task_embedding_attn_merge_type = self.hparams.task_embedding_attn_merge_type

        if self.hparams.dataset_name=='uva_dar':
            self.task_activity_map = None
        elif self.hparams.dataset_name=='mit_ucsd':
            self.task_activity_map = config.mit_ucsd_task_activity

        self.mm_fuser = nn.ModuleDict()
        
        if self.hparams.task_fusion_attention_type=='multi_head':
            self.mm_fuser = nn.MultiheadAttention(embed_dim=self.modality_embedding_size,
                                                num_heads=self.multi_modal_nhead,
                                                dropout=self.dropout)
        elif self.hparams.task_fusion_attention_type=='keyless':
            self.mm_fuser = KeylessAttention(self.modality_embedding_size)
        
        self.mm_mhattn_relu = nn.ReLU()
        self.mm_mhattn_dropout = nn.Dropout(p=self.dropout)

        mm_input_dim = 2*self.modality_embedding_size
        mm_output_dim = mm_input_dim // 2
        self.fc_output1 = nn.Linear(mm_input_dim, mm_output_dim)
        self.har_classifier = nn.Linear(mm_output_dim, 
                                    self.hparams.total_activities)

        self.fc_output1.apply(self.init_weights)                            
        self.har_classifier.apply(self.init_weights)
    
    def forward(self, module_out, task_embed): 
        
        mm_embeddings = []
        for modality in self.modalities:
            mm_embeddings.append(module_out[modality])
        
        mm_embeddings = torch.stack(mm_embeddings, dim=1).contiguous()
        mm_embeddings = F.relu(mm_embeddings)
        nbatches = mm_embeddings.shape[0]

        if self.num_modality>1 and self.hparams.task_fusion_attention_type is not None:
            if self.hparams.task_fusion_attention_type=='multi_head':
                
                # transpose batch and sequence (B x S x ..) --> (S x B x ..)
                mm_embeddings = mm_embeddings.transpose(0, 1).contiguous()

                query = mm_embeddings
                if task_embed is not None:
                    query = task_embed.unsqueeze(dim=0).contiguous()
                    # task_embed = task_embed.transpose(0, 1).contiguous()

                # remove the modality mask batch['modality_mask']
                # so that the model can learn by itself that 
                # how it can learn in the presence of missing modality
                mm_embeddings, self.mm_attn_weight = self.mm_fuser(query, 
                                                                mm_embeddings, 
                                                                mm_embeddings)
                # transpose batch and sequence (S x B x ..) --> (B x S x ..)
                mm_embeddings = mm_embeddings.transpose(0, 1).contiguous()  
                
            elif self.hparams.task_fusion_attention_type=='keyless':
                mm_embeddings, self.mm_attn_weight = self.mm_fuser(mm_embeddings)

            mm_embeddings = self.mm_mhattn_dropout(self.mm_mhattn_relu(mm_embeddings))
        
        mm_embeddings = mm_embeddings.contiguous().view(nbatches, -1)
        mm_embed = torch.cat([mm_embeddings, task_embed], 1)
        mm_embed = F.relu(self.fc_output1(mm_embed))

        logits = self.har_classifier(mm_embed)
        return logits

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
