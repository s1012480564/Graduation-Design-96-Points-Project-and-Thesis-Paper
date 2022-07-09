# -*- coding: utf-8 -*-
# file: td_lstm.py
# author: songanyang <1012480564@qq.com>
# Copyright (C) 2022. All Rights Reserved.

import torch
import torch.nn as nn


class TDLC_Transformer(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TDLC_Transformer, self).__init__()
        self.embed_dim = opt.embed_dim
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_l = nn.LSTM(opt.embed_dim, opt.hidden_dim, 1, batch_first=True)
        self.lstm_r = nn.LSTM(opt.embed_dim, opt.hidden_dim, 1, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=opt.embed_dim, nhead=6, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=opt.coder_num_layers)
        self.feed_forward_conv = nn.Conv1d(3 * opt.max_seq_len, 1, 1)
        self.dense = nn.Linear(3*opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        x_l, x_r, x = inputs[0], inputs[1], inputs[2]
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_l, x_r, x = self.embed(x_l), self.embed(x_r), self.embed(x)
        x_l = self.lstm_l(x_l)[0]
        x_r = self.lstm_r(x_r)[0]
        x_en = torch.cat((x_l, x_r, x), dim=1)
        x_en = self.encoder(x_en)
        h_n_en = self.feed_forward_conv(x_en)
        h_n_en = h_n_en.reshape(-1,self.embed_dim)
        h_n_l, h_n_r = torch.empty_like(x_l[:, 0, :]), torch.empty_like(x_r[:, 0, :])
        for i in range(x_l_len.size(0)):
            h_n_l[i, :] = x_l[i, x_l_len[i] - 1]
            h_n_r[i, :] = x_r[i, x_r_len[i] - 1]
        h_n = torch.cat((h_n_l, h_n_r, h_n_en), dim=-1)
        out = self.dense(h_n)
        return out