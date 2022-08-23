#!/usr/bin/env python
# -*- coding:utf-8 -*-

from model.BasiclLayer import BasicCTR, BasicCL4CTR, FactorizationMachine, MultiLayerPerceptron


class DeepFM(BasicCTR):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DeepFM, self).__init__(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_size = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size, mlp_layers, dropout, output_layer=True)

    def forward(self, x):
        """
        :param x: B,F
        :return:
        """
        x_embed = self.embedding(x)  # B,F,E
        x_out = self.lr(x) + self.fm(x_embed) + self.mlp(x_embed.view(x.size(0), -1))
        return x_out


class DeepFM_CL4CTR(BasicCL4CTR):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5, batch_size=1024, pratio=0.5,
                 fi_type="att"):
        super(DeepFM_CL4CTR, self).__init__(field_dims, embed_dim, batch_size, pratio=pratio, fi_type=fi_type)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_size = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_size, mlp_layers, dropout, output_layer=True)

    def forward(self, x):
        """
        :param x: B,F
        :return:
        """
        x_embed = self.embedding(x)  # B,F,E
        x_out = self.lr(x) + self.fm(x_embed) + self.mlp(x_embed.view(x.size(0), -1))
        return x_out
