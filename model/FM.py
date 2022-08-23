#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasiclLayer import BasicCTR, FeaturesLinear, FactorizationMachine, FeaturesEmbedding
from model.FRNet import FRNet


class FactorizationMachineModel(BasicCTR):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super(FactorizationMachineModel, self).__init__(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: B,F
        :return:
        """
        emb_x = self.embedding(x)
        x = self.lr(x) + self.fm(emb_x)
        return x


class FMFRNet(BasicCTR):
    """
    FM with FRNet
    """
    def __init__(self, field_dims, embed_dim,
                 num_layers=1, weight_type="bit", att_size=20, mlp_layer=256):
        super(FMFRNet, self).__init__(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.num_fields = len(field_dims)
        self.frnet = FRNet(self.num_fields, embed_dim, weight_type=weight_type,
                           num_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

    def forward(self, x):
        """
        :param x: B,F
        :return:
        """
        emb_x = self.embedding(x)
        # Applying Feature refinement to learn context-aware feature representations
        emb_x = self.frnet(emb_x)

        x = self.lr(x) + self.fm(emb_x)
        return x
        # adjust output according training file.
        # return F.sigmoid(x)


if __name__ == '__main__':
    import numpy as np

    fd = [3, 4]
    embed_dim = 8
    f_n = np.array([[1, 3], [0, 2], [0, 1], [1, 3]])
    f_n = torch.from_numpy(f_n).long()
    model = FMFRNet(fd, embed_dim)
    label = torch.randint(0, 2, (4, 1)).float()
    print(label)
    loss = nn.BCEWithLogitsLoss()
    pred = model(f_n)
    print(pred.size())
