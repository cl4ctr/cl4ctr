#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasiclLayer import BasicCTR, FeaturesLinear, FactorizationMachine, \
    FeaturesEmbedding, MultiLayerPerceptron
from model.FRNet import FRNet

class DeepFM(BasicCTR):
    def __init__(self,field_dims,embed_dim,mlp_layers=(400,400,400),dropout=0.5):
        super(DeepFM, self).__init__(field_dims,embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_size = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size,mlp_layers,dropout,output_layer=True)

    def forward(self,x):
        """
        :param x: B,F
        :return:
        """
        x_embed = self.embedding(x)  # B,F,E
        x_out = self.lr(x) + self.fm(x_embed) + self.mlp(x_embed.view(x.size(0),-1))
        return x_out


class DeepFM_FRNet(BasicCTR):
    def __init__(self,field_dims,embed_dim,mlp_layers=(400,400,400),dropout=0.5,
                 num_layers=1, weight_type="bit", att_size=20, mlp_layer=256):
        super(DeepFM_FRNet, self).__init__(field_dims,embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_size = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size,mlp_layers,dropout,output_layer=True)

        self.frnet = FRNet(self.num_fields, embed_dim, weight_type=weight_type,
                           num_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

    def forward(self,x):
        """
        :param x: B,F
        :return:
        """
        x_embed = self.embedding(x)  # B,F,E
        x_embed = self.frnet(x_embed)
        x_out = self.lr(x) + self.fm(x_embed) + self.mlp(x_embed.view(x.size(0),-1))
        return x_out