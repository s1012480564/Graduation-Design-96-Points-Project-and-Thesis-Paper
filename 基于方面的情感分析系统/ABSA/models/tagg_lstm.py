# -*- coding: utf-8 -*-
# file: td_lstm.py
# author: songanyang <1012480564@qq.com>
# Copyright (C) 2022. All Rights Reserved.

import torch
import torch.nn as nn
from layers.dynamic_rnn import DynamicLSTM


class TAGG_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TAGG_LSTM, self).__init__()
        self.device = opt.device
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_ls = [nn.LSTM(opt.embed_dim, opt.hidden_dim, 1, batch_first=True).to(opt.device) for i in range(4)]
        self.lstm_rs = [nn.LSTM(opt.embed_dim, opt.hidden_dim, 1, batch_first=True).to(opt.device) for i in range(4)]
        self.dense = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        x_l, x_r = inputs[0], inputs[1]
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        for k in range(4):
            mask_l, mask_r = torch.ones_like(x_l), torch.ones_like(x_r)
            for i in range(x_l.size(0)):
                mask_l[i, 0:int((x_l_len[i]-6)*(k+1)*0.25)] = 0
                mask_r[i, 0:int((x_r_len[i]-6)*(k+1)*0.25)] = 0
            tmp_l, tmp_r = self.lstm_ls[k](x_l)[0],self.lstm_ls[k](x_r)[0]
            tmp_l = torch.mul(tmp_l, mask_l)
            tmp_r = torch.mul(tmp_r, mask_r)
            x_l = x_l + tmp_l
            x_r = x_r + tmp_r
        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out
