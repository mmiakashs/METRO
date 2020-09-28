import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Vis_Module(nn.Module):
    def __init__(self, feature_embed_size, lstm_hidden_size,
                 fine_tune=False, kernel_size=3, cnn_in_channel=3,
                 batch_first=True, window_size=1, window_stride=1, n_head=4,
                 dropout=0.6, activation="relu", encoder_num_layers=2,
                 pool_fe_kernel=None, pool_fe_stride=None, pool_fe_type='max',
                 lstm_bidirectional=False, lstm_dropout=0.1,
                 adaptive_pool_tar_squze_mul=None,
                 attention_type='mm',
                 is_attention=True,
                 is_pretrained_fe = False):

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
        self.window_size = window_size
        self.window_stride = window_stride
        self.encoder_num_layers = encoder_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.pool_fe_kernel = pool_fe_kernel
        self.pool_fe_stride = pool_fe_stride
        self.pool_fe_type = pool_fe_type
        self.adaptive_pool_tar_squze_mul = adaptive_pool_tar_squze_mul
        self.attention_type = attention_type
        self.is_attention = is_attention
        self.is_pretrained_fe = is_pretrained_fe

        if(self.cnn_in_channel==1):
            self.feature_extractor = models.resnet34(pretrained=True)
            self.feature_extractor.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            self.feature_extractor = models.resnet50(pretrained=True)

        num_ftrs = self.feature_extractor.fc.in_features
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

        if (self.is_attention):
            if (self.lstm_bidirectional):
                self.self_attn = nn.MultiheadAttention(embed_dim=2 * self.feature_embed_size,
                                                           num_heads=n_head,
                                                           dropout=self.dropout)
            else:
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

    def forward(self, input, input_mask, input_len):
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
            print(f'###### type {type(embed)}')
            embed = self.fe_fc(embed)

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
        r_output, (h_n, h_c) = self.lstm(embed)

        if (self.is_attention):
            input_mask = input_mask[:, :r_output.size(1)]
            r_output = r_output.transpose(0,1).contiguous()

            attn_output, self.self_attn_weight = self.self_attn(r_output, r_output, r_output,
                                                                key_padding_mask=input_mask)
            attn_output = attn_output.transpose(0,1).contiguous()
            attn_output = torch.sum(attn_output, dim=1).squeeze(dim=1)
            attn_output = F.relu(attn_output)
            attn_output = self.module_fe_dropout(attn_output)

        else:
            attn_output = r_output[:, -1,:]

        return attn_output, self.self_attn_weight
