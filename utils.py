
import math, torch

import torch
from torch.nn.utils.rnn import pad_sequence


def pad_from_mask(feats, mask):
    B, max_len = mask.shape
    _, d_model = feats.shape
    
    lengths = mask.sum(dim=1).long().tolist()
    pad_feats, start = [], 0
    for seq_len in lengths:
        end = start + seq_len
        pad_len = max_len - seq_len
        pad_tensor = torch.zeros(pad_len, d_model, device=feats.device)
        feat = torch.cat([feats[start:end], pad_tensor], dim=0)
        pad_feats.append(feat)
        start = end

    return torch.stack(pad_feats, dim=0)

def unpad_from_mask(pad_feats, mask):
    lengths = mask.sum(dim=1).long().tolist()
    feats_list = [pad_feats[i, :l] for i, l in enumerate(lengths)]
    return torch.cat(feats_list, dim=0)



def extract_pocket_features_batch(feat, pocket_mask, max_len):
    B, L, D = feat.size()
    pocket_list = [feat[b][pocket_mask[b]] for b in range(B)]

    pocket_clipped = [p[:max_len] for p in pocket_list]

    padded = pad_sequence(pocket_clipped, batch_first=True)
    mask = torch.ones(padded.size()[:2], dtype=torch.bool, device=feat.device)

    if padded.size(1) < max_len:
        pad_w = max_len - padded.size(1)
        padded = torch.cat([padded, torch.zeros(B, pad_w, D, device=feat.device)], dim=1)
        mask = torch.cat([mask, torch.zeros(B, pad_w, dtype=torch.bool, device=feat.device)], dim=1)

    return padded, mask





import torch, math

class NoamOpt:
    def __init__(self, d_model, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.factor = factor
        self.warmup = warmup
        self.d_model = d_model

    def step(self):
        self._step += 1
        lr = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = lr
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.d_model ** -0.5 *
                              min(step ** -0.5, step * self.warmup ** -1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(len_size, batch_size, parameters, d_model, top_lr=5e-4):
    step_each_epoch = math.ceil(len_size / batch_size)
    warmup = 8 * step_each_epoch
    factor = top_lr * math.sqrt(warmup)
    opt = torch.optim.AdamW(parameters, lr=0,
                            betas=(0.9, 0.98),
                            eps=1e-9,
                            weight_decay=0.001)
    return NoamOpt(d_model, factor, warmup, opt)