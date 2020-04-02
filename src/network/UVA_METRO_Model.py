import torch
import torch.nn as nn
import torch.nn.functional as F

from .Vis_Module import Vis_Module
from ..utils import config


class UVA_METRO_Model(nn.Module):
    def __init__(self, mm_module_properties,
                 modalities,
                 num_activity_types,
                 window_size, window_stride,
                 modality_embedding_size,
                 batch_first=True,
                 mm_embedding_attn_merge_type='sum',
                 dropout=0.1,
                 activation="relu",
                 nn_init_type='xu',
                 is_pretrained_fe=False):
        super(UVA_METRO_Model, self).__init__()

        self.mm_module_properties = mm_module_properties
        self.modalities = modalities
        self.num_modality = len(modalities)
        self.num_activity_types = num_activity_types
        self.batch_first = batch_first

        self.mm_embedding_attn_merge_type = mm_embedding_attn_merge_type
        self.dropout = dropout
        self.activation = activation
        self.window_size = window_size
        self.window_stride = window_stride
        self.lstm_bidirectional = False
        self.modality_embedding_size = modality_embedding_size
        self.nn_init_type = nn_init_type
        self.is_pretrained_fe = is_pretrained_fe

        self.mm_module = nn.ModuleDict()
        for modality in self.modalities:
            self.mm_module[modality] = Vis_Module(cnn_in_channel=self.mm_module_properties[modality]['cnn_in_channel'],
                                                  feature_embed_size=self.mm_module_properties[modality][
                                                      'feature_embed_size'],
                                                  kernel_size=self.mm_module_properties[modality]['kernel_size'],
                                                  lstm_hidden_size=self.mm_module_properties[modality][
                                                      'lstm_hidden_size'],
                                                  fine_tune=self.mm_module_properties[modality]['fine_tune'],
                                                  batch_first=self.batch_first,
                                                  window_size=self.window_size,
                                                  window_stride=self.window_stride,
                                                  n_head=self.mm_module_properties[modality]['module_embedding_nhead'],
                                                  dropout=self.mm_module_properties[modality]['dropout'],
                                                  activation=self.mm_module_properties[modality]['activation'],
                                                  encoder_num_layers=self.mm_module_properties[modality][
                                                      'lstm_encoder_num_layers'],
                                                  lstm_bidirectional=self.mm_module_properties[modality][
                                                      'lstm_bidirectional'],
                                                  lstm_dropout=self.mm_module_properties[modality]['lstm_dropout'],
                                                  pool_fe_kernel=self.mm_module_properties[modality][
                                                      'feature_pooling_kernel'],
                                                  pool_fe_stride=self.mm_module_properties[modality][
                                                      'feature_pooling_stride'],
                                                  pool_fe_type=self.mm_module_properties[modality][
                                                      'feature_pooling_type'],
                                                  is_attention = self.mm_module_properties[modality][
                                                      'is_attention'],
                                                  is_pretrained_fe = self.is_pretrained_fe)

            if (self.mm_module_properties[modality]['lstm_bidirectional']):
                self.lstm_bidirectional = True

        if (self.lstm_bidirectional):
            self.modality_embedding_size = 2 * self.modality_embedding_size

        self.mm_embeddings_bn =nn.BatchNorm1d(self.num_modality)

        if (self.mm_embedding_attn_merge_type == 'sum'):
            if (self.lstm_bidirectional):
                self.fc_output1 = nn.Linear(self.modality_embedding_size, self.modality_embedding_size // 2)
                self.fc_output2 = nn.Linear(self.modality_embedding_size // 2, self.modality_embedding_size // 4)
                self.fc_output3 = nn.Linear(self.modality_embedding_size // 4, self.num_activity_types)
            else:
                self.fc_output1 = nn.Linear(self.modality_embedding_size, self.modality_embedding_size // 2)
                self.fc_output2 = nn.Linear(self.modality_embedding_size // 2, self.num_activity_types)
        else:
            if (self.lstm_bidirectional):
                self.fc_output1 = nn.Linear(self.num_modality * self.modality_embedding_size,
                                            self.num_modality * self.modality_embedding_size // 2)
                self.fc_output2 = nn.Linear(self.num_modality * self.modality_embedding_size // 2,
                                            self.num_modality * self.modality_embedding_size // 4)
                self.fc_output3 = nn.Linear(self.num_modality * self.modality_embedding_size // 4,
                                            self.num_activity_types)
            else:
                self.fc_output1 = nn.Linear(self.num_modality * self.modality_embedding_size,
                                            self.num_modality * self.modality_embedding_size // 2)
                self.fc_output2 = nn.Linear(self.num_modality * self.modality_embedding_size // 2,
                                            self.num_activity_types)

        self.module_attn_weights = {}
        self.mm_attn_weight = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_output1.weight)
        nn.init.constant_(self.fc_output1.bias, 0.)

        nn.init.xavier_uniform_(self.fc_output2.weight)
        nn.init.constant_(self.fc_output2.bias, 0.)

        if (self.lstm_bidirectional):
            nn.init.xavier_uniform_(self.fc_output3.weight)
            nn.init.constant_(self.fc_output3.bias, 0.)

    def forward(self, input):
        mm_module_output = {}
        for modality in self.modalities:
            tm_attn_output, self.module_attn_weights[modality] = self.mm_module[modality](input[modality],
                                                                                          input[modality + config.modality_mask_suffix_tag],
                                                                                          input[modality + config.modality_seq_len_tag])
            mm_module_output[modality] = tm_attn_output

        mm_embeddings = []
        for tm_modality in mm_module_output.keys():
            mm_embeddings.append(mm_module_output[tm_modality])

        mm_embeddings = torch.stack(mm_embeddings, dim=1).contiguous()
        mm_embeddings = F.relu(self.mm_embeddings_bn(mm_embeddings))
        nbatches = mm_embeddings.shape[0]

        if(self.mm_embedding_attn_merge_type=='sum'):
            mattn_output = torch.sum(mm_embeddings, dim=1).squeeze(dim=1)

        mattn_output = mm_embeddings.contiguous().view(nbatches, -1).contiguous()

        if (self.lstm_bidirectional):
            output = self.fc_output1(mattn_output)
            output = self.fc_output2(output)
            output = self.fc_output3(output)
        else:
            output = self.fc_output1(mattn_output)
            output = self.fc_output2(output)

        return F.log_softmax(output, dim=1)
