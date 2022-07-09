# -*- coding: utf-8 -*-
# file: td_lstm.py
# author: songanyang <1012480564@qq.com>
# Copyright (C) 2022. All Rights Reserved.

import torch
import torch.nn as nn
import math


class TD_Transformer(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TD_Transformer, self).__init__()
        self.mask = torch.triu(torch.ones(opt.max_seq_len, opt.max_seq_len) * float('-inf'), diagonal=1).to(opt.device)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.encoder_l = nn.TransformerEncoder(encoder_layer, num_layers=opt.coder_num_layers)
        self.encoder_r = nn.TransformerEncoder(encoder_layer, num_layers=opt.coder_num_layers)
        self.dense = nn.Linear(512*2, opt.polarities_dim)

        position = torch.arange(opt.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 512-opt.embed_dim, 2) * (-math.log(10000.0) / (512-opt.embed_dim)))
        self.pe = torch.zeros(opt.batch_size,opt.max_seq_len, 512-opt.embed_dim)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.to(opt.device)

    def forward(self, inputs):
        x_l, x_r = inputs[0], inputs[1]
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        x_l, x_r = torch.cat([x_l,self.pe[:x_l.size(0)]],-1), torch.cat([x_r,self.pe[:x_r.size(0)]],-1)
        x_l, x_r = self.encoder_l(x_l, mask=self.mask), self.encoder_r(x_r, mask=self.mask)
        h_n_l, h_n_r = torch.empty_like(x_l[:, 0, :]), torch.empty_like(x_r[:, 0, :])
        for i in range(x_l_len.size(0)):
            h_n_l[i, :] = x_l[i, x_l_len[i] - 1]
            h_n_r[i, :] = x_r[i, x_r_len[i] - 1]
        x = torch.cat((h_n_l, h_n_r), dim=-1)
        out = self.dense(x)
        return out
