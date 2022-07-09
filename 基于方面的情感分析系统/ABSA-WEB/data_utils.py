# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songanyang <1012480564@qq.com>
# Copyright (C) 2022. All Rights Reserved.

import numpy as np
import torch
from transformers import BertTokenizer


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


def get_process_input(text, asp_l, asp_r, tokenizer, opt):
    text = text.lower().strip()
    asp_words = text.split()[asp_l:asp_r + 1]
    aspect = asp_words[0]
    for i in range(1, len(asp_words)):
        aspect += ' ' + asp_words[i]

    text_indices = tokenizer.text_to_sequence(text)
    aspect_indices = tokenizer.text_to_sequence(aspect)
    aspect_len = np.sum(aspect_indices != 0)
    text_len = np.sum(text_indices != 0)

    concat_bert_indices = torch.tensor(
        [tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + aspect + ' [SEP]')]).to(opt.device)

    concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
    concat_segments_indices = torch.tensor([pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)]).to(
        opt.device)

    text_bert_indices = torch.tensor([tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP]')]).to(opt.device)

    aspect_bert_indices = torch.tensor([tokenizer.text_to_sequence('[CLS] ' + aspect + ' [SEP]')]).to(opt.device)

    return [concat_bert_indices, concat_segments_indices, text_bert_indices, aspect_bert_indices]


