#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class FRNet(nn.Module):
    """
    Feature refinement network：
    (1) IEU
    (2) CSGate
    """
    def __init__(self, field_length, embed_dim, weight_type="bit", num_layers=1, att_size=10, mlp_layer=256):
        """
        :param field_length: field_length
        :param embed_dim: embedding dimension
        type: bit or vector
        """
        super(FRNet, self).__init__()
        # IEU_G computes complementary features.
        self.IEU_G = IEU(field_length, embed_dim, weight_type="bit",
                         bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

        # IEU_W computes bit-level or vector-level weights.
        self.IEU_W = IEU(field_length, embed_dim, weight_type=weight_type,
                         bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)

    def forward(self, x_embed):
        com_feature = self.IEU_G(x_embed)
        wegiht_matrix = torch.sigmoid(self.IEU_W(x_embed))
        # CSGate
        x_out = x_embed * wegiht_matrix + com_feature * (torch.tensor(1.0) - wegiht_matrix)
        return x_out


class IEU(nn.Module):
    """
    Information extraction Unit (IEU) for FRNet
    (1) Self-attention
    (2) DNN
    """
    def __init__(self, field_length, embed_dim, weight_type="bit",
                 bit_layers=1, att_size=10, mlp_layer=256):
        """
        :param field_length:
        :param embed_dim:
        :param type: vector or bit
        :param bit_layers:
        :param att_size:
        :param mlp_layer:
        """
        super(IEU,self).__init__()
        self.input_dim = field_length * embed_dim
        self.weight_type = weight_type

        # Self-attention unit, which is used to capture cross-feature relationships.
        self.vector_info = SelfAttentionIEU(embed_dim=embed_dim, att_size=att_size)

        #  contextual information extractor(CIE), we adopt MLP to encode contextual information.
        mlp_layers = [mlp_layer for _ in range(bit_layers)]
        self.mlps = MultiLayerPerceptronPrelu(self.input_dim, embed_dims=mlp_layers,
                                              output_layer=False)
        self.bit_projection = nn.Linear(mlp_layer, embed_dim)
        self.activation = nn.ReLU()
        # self.activation = nn.PReLU()


    def forward(self,x_emb):
        """
        :param x_emb: B,F,E
        :return: B,F,E (bit-level weights or complementary fetures)
                 or B,F,1 (vector-level weights)
        """

        # （1）self-attetnion unit
        x_vector = self.vector_info(x_emb)  # B,F,E

        # (2) CIE unit
        x_bit = self.mlps(x_emb.view(-1, self.input_dim))
        x_bit = self.bit_projection(x_bit).unsqueeze(1) # B,1,e
        x_bit = self.activation(x_bit)

        # （3）integration unit
        x_out = x_bit * x_vector

        if self.weight_type == "vector":
            # To compute vector-level importance in IEU_W
            x_out = torch.sum(x_out,dim=2,keepdim=True)
            # B,F,1
            return x_out

        return x_out


class SelfAttentionIEU(nn.Module):
    def __init__(self, embed_dim, att_size=20):
        """
        :param embed_dim:
        :param att_size:
        """
        super(SelfAttentionIEU, self).__init__()
        self.embed_dim = embed_dim
        self.trans_Q = nn.Linear(embed_dim,att_size)
        self.trans_K = nn.Linear(embed_dim,att_size)
        self.trans_V = nn.Linear(embed_dim,att_size)
        self.projection = nn.Linear(att_size,embed_dim)
        # self.scale = 1.0/ torch.LongTensor(embed_dim)
        # self.scale = torch.sqrt(1.0 / torch.tensor(embed_dim).float())
        # self.dropout = nn.Dropout(0.5)
        # self.layer_norm = nn.LayerNorm(embed_dim)


    def forward(self,x, scale=None):
        """
        :param x: B,F,E
        :return: B,F,E
        """
        Q = self.trans_Q(x)
        K = self.trans_K(x)
        V = self.trans_V(x)
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # B,F,F
        attention_score = F.softmax(attention, dim=-1)
        context = torch.matmul(attention_score, V)
        # Projection
        context = self.projection(context)
        # context = self.layer_norm(context)
        return context


class MultiLayerPerceptronPrelu(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.PReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        :param x: [B,F*E]
        """
        return self.mlp(x)


if __name__ == '__main__':
    x_emb = torch.randn(32,10,20)
    print(x_emb.size())
    frnet = FRNet(10,20)
    x_emb2 = frnet(x_emb)
    print(x_emb2.size())
