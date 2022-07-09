# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songanyang <1012480564@qq.com>
# Copyright (C) 2022. All Rights Reserved.

import torch
from transformers import BertModel
from data_utils import Tokenizer4Bert, get_process_input
from models import LCF_BERT
import argparse


def predict(text, asp_l, asp_r):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--local_context_focus', default='cdw', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=5, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')

    opt = parser.parse_args(args=[])

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
    model = LCF_BERT(bert, opt).to(opt.device)

    model.load_state_dict(torch.load('state_dict/lcf_bert_twitter_val_acc_0.7442'))

    model.eval()

    input = get_process_input(text, asp_l, asp_r, tokenizer, opt)

    with torch.no_grad():
        output = model(input)
        polarity = torch.argmax(output[0]).item() - 1

    return polarity


def get_weights(text_words, asp_l, asp_r):
    SRD = 5
    text_len = len(text_words)
    weights = [[i,0,1 if min(abs(i-asp_l),abs(i-asp_r))<=SRD else (text_len-min(abs(i-asp_l),abs(i-asp_r)))/text_len] for i in range(text_len)]
    return weights
