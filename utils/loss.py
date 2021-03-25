# coding: utf-8
# @file: loss.py

import torch
import torch.nn as nn

import sys
import datetime


class CategoricalLoss(nn.Module):
    def __init__(self, atoms=51, v_max=10, v_min=-10):
        super(CategoricalLoss, self).__init__()

        self.atoms = atoms
        self.v_max = v_max
        self.v_min = v_min
        self.supports = torch.linspace(v_min, v_max, atoms).view(1, 1, atoms) # RL: [bs, #action, #quantiles]
        self.delta = (v_max - v_min) / (atoms - 1)

    def to(self, device):
        self.device = device
        self.supports = self.supports.to(device)

    def forward(self, anchor, feature, skewness=0.0, direction=None, weight=None):
        batch_size = feature.shape[0]
        if direction is not None:
            skew = torch.zeros((batch_size, self.atoms)).to(self.device)
            for i, s in enumerate(skew):
                if direction[i] == 1:
                    skew[i].fill_(-skewness)
                else:
                    skew[i].fill_(skewness)
        else:
            skew = torch.zeros((batch_size, self.atoms)).to(self.device).fill_(skewness)

        # experiment to adjust KL divergence between positive/negative anchors
        Tz = skew + self.supports.view(1, -1) * torch.ones((batch_size, 1)).to(torch.float).view(-1, 1).to(self.device)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta
        l = b.floor().to(torch.int64)
        u = b.ceil().to(torch.int64)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1
        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).to(torch.int64).unsqueeze(dim=1).expand(batch_size, self.atoms).to(self.device)
        skewed_anchor = torch.zeros(batch_size, self.atoms).to(self.device)
        skewed_anchor.view(-1).index_add_(0, (l + offset).view(-1), (anchor * (u.float() - b)).view(-1))
        skewed_anchor.view(-1).index_add_(0, (u + offset).view(-1), (anchor * (b - l.float())).view(-1))

        if weight is not None:
            loss = -((skewed_anchor * (feature + 1e-16).log()).sum(-1) * weight).mean()
        else:
            loss = -(skewed_anchor * (feature + 1e-16).log()).sum(-1).mean()

        return loss