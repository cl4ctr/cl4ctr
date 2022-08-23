#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasiclLayer import BasicCTR, FeaturesLinear, FactorizationMachine, FeaturesEmbedding
from model.FRNet import FRNet


class FactorizationMachineModel(BasicCTR):
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
    def __init__(self, field_dims, embed_dim,
                 num_layers=1, weight_type="bit", att_size=20, mlp_layer=256):
        super(FMFRNet, self).__init__(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.num_fields = len(field_dims)
        self.frnet = FRNet(self.num_fields, embed_dim, weight_type=weight_type,
                           num_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

    def forward(self, x):
        emb_x = self.embedding(x)
        emb_x = self.frnet(emb_x)

        x = self.lr(x) + self.fm(emb_x)
        return x