import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.models.KeylessAttention import KeylessAttention


class Vis_Module(nn.Module):
    def __init__(self, feature_embed_size, lstm_hidden_size,
                 fine_tune=False, kernel_size=3, cnn_in_channel=3,
                 batch_first=True, n_head=4,
                 dropout=0.1, activation="relu", encoder_num_layers=2, lstm_bidirectional=False,
                 pool_fe_kernel=None, pool_fe_stride=None, pool_fe_type='max', lstm_dropout=0.1,
                 adaptive_pool_tar_squze_mul=None,
                 attention_type='muti_head', is_attention=True,
                 is_pretrained_fe = False,
                 pt_vis_encoder_archi_type='resnet50'):

        super(Vis_Module, self).__init__()

        self.cnn_in_channel = cnn_in_channel
        self.feature_embed_size = feature_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.fine_tune = fine_tune
        self.batch_first = batch_first
        self.kernel_size = kernel_size
        self.n_head = n_head
        self.dropout = dropout
        self.lstm_dropout = lstm_dropout
        self.activation = activation
        self.encoder_num_layers = encoder_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.pool_fe_kernel = pool_fe_kernel
        self.pool_fe_stride = pool_fe_stride
        self.pool_fe_type = pool_fe_type
        self.adaptive_pool_tar_squze_mul = adaptive_pool_tar_squze_mul
        self.attention_type = attention_type
        self.is_attention = is_attention
        self.is_pretrained_fe = is_pretrained_fe
        self.pt_vis_encoder_archi_type = pt_vis_encoder_archi_type

        if(self.cnn_in_channel==1):
            self.feature_extractor = models.resnet18(pretrained=True)
            self.feature_extractor.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            if self.pt_vis_encoder_archi_type=='resnet18':
                self.feature_extractor = models.resnet18(pretrained=True)
            elif self.pt_vis_encoder_archi_type=='resnet34':
                self.feature_extractor = models.resnet34(pretrained=True)
            else:
                self.feature_extractor = models.resnet50(pretrained=True)
        
        # I add this work around as I used resnet50 pretrained pre-extracted features.
        # however, later I wanted to use other resnet models(e.g. resnet18/34)
        # these resnet models output dimension don't match the resnet50 extracted
        # features dimension. In general we can just used the else part
        if self.pt_vis_encoder_archi_type!='resnet50' and self.is_pretrained_fe:
            num_ftrs = 2048
        else:
            num_ftrs = self.feature_extractor.fc.in_features
        # end of the work around condition to used the pretrained features

        self.feature_extractor.fc = nn.Linear(num_ftrs, self.feature_embed_size)
        if (self.fine_tune):
            self.set_parameter_requires_grad(self.feature_extractor, self.fine_tune)

        self.fe_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fe_fc = nn.Linear(num_ftrs, self.feature_embed_size)

        self.fe_relu = nn.ReLU()
        self.fe_dropout = nn.Dropout(p=self.dropout)
        if(self.pool_fe_kernel):
            if(self.pool_fe_type=='max'):
                self.pool_fe = nn.MaxPool1d(kernel_size=self.pool_fe_kernel,
                                        stride=self.pool_fe_stride)
            else:
                self.pool_fe = nn.AvgPool1d(kernel_size=self.pool_fe_kernel,
                                            stride=self.pool_fe_stride)

        self.lstm = nn.LSTM(input_size=self.feature_embed_size,
                            hidden_size=self.lstm_hidden_size,
                            batch_first=self.batch_first,
                            num_layers=self.encoder_num_layers,
                            bidirectional=self.lstm_bidirectional,
                            dropout=self.lstm_dropout)

        if self.lstm_bidirectional:
            if self.attention_type == 'keyless':
                self.self_attn = KeylessAttention(2 * self.feature_embed_size)
            elif self.attention_type == 'multi_head':
                self.self_attn = nn.MultiheadAttention(embed_dim=2 * self.feature_embed_size,
                                                        num_heads=n_head,
                                                        dropout=self.dropout)
        else:
            if self.attention_type == 'keyless':
                self.self_attn = KeylessAttention(self.feature_embed_size)
            elif self.attention_type == 'multi_head':
                self.self_attn = nn.MultiheadAttention(embed_dim=self.feature_embed_size,
                                                        num_heads=n_head,
                                                        dropout=self.dropout)

        self.self_attn_weight = None
        self.module_fe_relu = nn.ReLU()
        self.module_fe_dropout = nn.Dropout(p=self.dropout)

    def set_parameter_requires_grad(self, model, fine_tune):
        for param in model.parameters():
            param.requires_grad = self.fine_tune
                
    def gen_mask(self, seq_len, max_len):
        return torch.arange(max_len) > seq_len

    def forward(self, input, 
                input_mask, input_len, 
                guided_context=None):
        # print('########### Start Vis_Module ###########')
        # print('input shape', input.size())

        if(not self.is_pretrained_fe):
            x = input.view(-1, input.size(-3), input.size(-2), input.size(-1)).contiguous()
            x = x.contiguous()

            embed = self.feature_extractor(x).contiguous()
            if (self.batch_first):
                embed = embed.contiguous().view(input.size(0), -1, embed.size(-1))
            else:
                embed = embed.view(-1, input.size(1), embed.size(-1))
            embed = self.fe_dropout(self.fe_relu(embed))
        else:
            x = input.view(-1, input.size(-3), input.size(-2), input.size(-1)).contiguous()
            embed = self.fe_pool(x)
            embed = embed.view(input.size(0), input.size(1), -1)
            embed = self.fe_fc(embed)
            embed = self.fe_dropout(self.fe_relu(embed))

        if(self.pool_fe_kernel):
            embed = embed.transpose(1,2).contiguous()
            embed = self.pool_fe(embed)
            embed = embed.transpose(1,2).contiguous()

        if (self.adaptive_pool_tar_squze_mul):
            embed = embed.transpose(1, 2).contiguous()
            adaptive_pool_tar_len = embed.shape[-1] // self.adaptive_pool_tar_squze_mul
            if (self.pool_fe_type == 'max'):
                embed = F.adaptive_max_pool1d(embed, adaptive_pool_tar_len)
            elif (self.pool_fe_type == 'avg'):
                embed = F.adaptive_avg_pool1d(embed, adaptive_pool_tar_len)
            embed = embed.transpose(1, 2).contiguous()

        self.lstm.flatten_parameters()

        # print('before sorting input len', input_len)
        # Sort by length (keep idx)
        # input_len = input_len.cpu().numpy()

        # input_len, idx_sort = torch.sort(input_len, descending=True)
        # idx_unsort = torch.argsort(idx_sort)
        # embed = embed.index_select(0, Variable(idx_sort))

        # Handling padding in Recurrent Networks
        # print('after sorting input len', input_len)
        # print('idx_sort', idx_sort)
        # print('idx_unsort', idx_unsort)
        # print('after sorting input len', input_len)
        # print('input embed size', embed.size())

        # input_packed = nn.utils.rnn.pack_padded_sequence(embed, input_len, batch_first=True)
        # r_output, (h_n, h_c) = self.lstm(input_packed)
        # r_output = nn.utils.rnn.pad_packed_sequence(r_output, batch_first=True)[0]
        #
        # r_output = r_output.index_select(0, Variable(idx_unsort))

        r_output, (h_n, h_c) = self.lstm(embed)

        if self.attention_type == 'multi_head':
            input_mask = input_mask[:, :r_output.size(1)]
            r_output = r_output.transpose(0, 1).contiguous() # transpose batch and sequence (B x S x ..) --> (S x B x ..)

            query = r_output
            if guided_context is not None:
                guided_context = guided_context.transpose(0, 1).contiguous()
                query = guided_context

            attn_output, self.self_attn_weight = self.self_attn(query, 
                                                        r_output, 
                                                        r_output, 
                                                        key_padding_mask=input_mask)
            attn_output = attn_output.transpose(0,1).contiguous()  # transpose batch and sequence (S x B x ..) --> (B x S x ..)
            attn_output = torch.sum(attn_output, dim=1).squeeze(dim=1)

        elif self.attention_type == 'keyless':
            attn_output, self.self_attn_weight = self.self_attn(r_output)
            attn_output = torch.sum(attn_output, 1).squeeze(1)
        else:
            attn_output = r_output[:,-1,:]
        
        attn_output = F.relu(attn_output)
        attn_output = self.module_fe_dropout(attn_output)

        # print('attn_output shape', attn_output.size())
        # print('########### End MM_Module ###########')

        return attn_output, self.self_attn_weight
