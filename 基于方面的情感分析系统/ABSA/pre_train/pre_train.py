import argparse
import logging
from time import strftime, localtime
import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop, all')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=768, type=int)
    parser.add_argument('--num_epoch', default=6, type=int, help='try 6 for each, 2 for all')
    parser.add_argument('--batch_size', default=4, type=int)
    opt = parser.parse_args()

    dataset_files = {
        'twitter': './datasets/twitter.txt',
        'restaurant': './datasets/restaurant.txt',
        'laptop': './datasets/laptop.txt',
        'all': './datasets/all.txt'
    }
    opt.dataset_path = dataset_files[opt.dataset]


    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    fin = open(opt.dataset_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = fin.readlines()

    tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name)
    model = BertForMaskedLM.from_pretrained(opt.pretrained_bert_name)

    class LineByLineTextDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = tokenizer.batch_encode_plus(data, add_special_tokens=True,
                                                        max_length=max_length, truncation=True)["input_ids"]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.tensor(self.data[idx], dtype=torch.long)

    dataset = LineByLineTextDataset(data, tokenizer, opt.max_seq_len)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    # dataloader = DataLoader(dataset, shuffle=True, batch_size=8, collate_fn=collate)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, collate_fn=collate)

    class Trainer:
        def __init__(self, model, dataloader, tokenizer, mlm_probability=0.15, lr=2e-5, with_cuda=True, cuda_devices=None,
                     log_freq=100):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model
            self.is_parallel = False
            self.dataloader = dataloader
            self.tokenizer = tokenizer
            self.mlm_probability = mlm_probability
            self.log_freq = log_freq

            # 多GPU训练
            if with_cuda and torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUS for BERT")
                self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
                self.is_parallel = True

            self.model.to(self.device)
            self.model.train()
            self.optim = AdamW(self.model.parameters(), lr=2e-5)
            # print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
            self._print_args()

        def _print_args(self):
            n_trainable_params, n_nontrainable_params = 0, 0
            for p in self.model.parameters():
                n_params = torch.prod(torch.tensor(p.shape))
                if p.requires_grad:
                    n_trainable_params += n_params
                else:
                    n_nontrainable_params += n_params
            logger.info('> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params,n_nontrainable_params))
            logger.info('> training arguments:')
            for arg in vars(opt):
                logger.info('>>> {0}: {1}'.format(arg, getattr(opt, arg)))

        def train(self, epoch):
            self.iteration(epoch, self.dataloader)

        def iteration(self, epoch, dataloader, train=True):
            str_code = 'Train'
            total_loss = 0.0
            for i, batch in tqdm(enumerate(dataloader), desc="Training"):
                inputs, labels = self._mask_tokens(batch)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                lm_loss, output = self.model(inputs, labels=labels, return_dict=False)
                loss = lm_loss.mean()

                if train:
                    self.model.zero_grad()
                    self.optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optim.step()

                total_loss += loss.item()
                post_fix = {
                    "iter": i,
                    "ave_loss": total_loss / (i + 1)
                }
                if i % self.log_freq == 0:
                    # print(post_fix)
                    logger.info(post_fix)

            # print(f"EP{epoch}_{str_code},avg_loss={total_loss / len(dataloader)}")
            logger.info(f"EP{epoch}_{str_code},avg_loss={total_loss / len(dataloader)}")

        def _mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """ Masked Language Model """
            if self.tokenizer.mask_token is None:
                raise ValueError(
                    "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
                )

            labels = inputs.clone()
            # 使用mlm_probability填充张量
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            # 获取special token掩码
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            # 将special token位置的概率填充为0
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            if self.tokenizer._pad_token is not None:
                # padding掩码
                padding_mask = labels.eq(tokenizer.pad_token_id)
                # 将padding位置的概率填充为0
                probability_matrix.masked_fill_(padding_mask, value=0.0)

            # 对token进行mask采样
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # loss只计算masked

            # 80%的概率将masked token替换为[MASK]
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10%的概率将masked token替换为随机单词
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
            inputs[indices_random] = random_words[indices_random]

            # 余下的10%不做改变
            return inputs, labels


    log_file = '{}-{}.log'.format(opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    trainer = Trainer(model, dataloader, tokenizer)

    for epoch in range(opt.num_epoch):
        trainer.train(epoch)

    model.save_pretrained("models/"+opt.dataset)

