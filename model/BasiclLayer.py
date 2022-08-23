#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BasicCTR(nn.Module):
    """
    A simple common-used class for CTR models.
    """
    def __init__(self,field_dims, embed_dim):
        """
        :param field_dims: list
        :param embed_dim:
        """
        super(BasicCTR, self).__init__()
        # linear part. most models need to use linear part.
        self.lr = FeaturesLinear(field_dims)
        # All CTR prediction models need to use embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x:  B,F,E
        :return:
        """
        raise NotImplemented

class FeaturesLinear(torch.nn.Module):
    """
    Linear regression layer for CTR prediction.
    """
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array(
            (0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: B,F
        :return: B,1
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: B,F,E
        """
        # 因为x都是1，所以不需要乘以x: 和的平方 - 平方的和
        square_of_sum = torch.sum(x, dim=1) ** 2  # B，embed_dim
        sum_of_square = torch.sum(x ** 2, dim=1)  # B，embed_dim
        # square of sum - sum of square
        ix = square_of_sum - sum_of_square  # B,embed_dim
        if self.reduce_sum:
            # For NFM, reduce_sum = False
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        """
        :param field_dims: list
        :param embed_dim
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self._init_weight_()

    def _init_weight_(self):
        """ weights initialization"""
        nn.init.normal_(self.embedding.weight, std=0.01)
        # nn.init.xavier_normal_nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        """
        :param x: B,F
        :return: B,F,E
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=False):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        # 使用 *，
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size,num_fields*embed_dim)``
        """
        return self.mlp(x)

class InnerProductNetwork(torch.nn.Module):

    def __init__(self,num_fields,is_sum=True):
        super(InnerProductNetwork, self).__init__()
        self.is_sum = is_sum
        self.num_fields = num_fields
        self.row, self.col = list(), list()

        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x):
        if self.is_sum == True:
            # 默认求和最原始的方式
            return torch.sum(x[:, self.row] * x[:, self.col], dim=2)  # B,1/2* nf*(nf-1)
        else:
            #  以下： 如果不求和 B,1/2* nf*(nf-1), K
            return x[:, self.row] * x[:, self.col]

class OuterProductNetwork(torch.nn.Module):
    def __init__(self, num_fields, embed_dim, kernel_type='num'):
        super().__init__()

        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type

        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))

        num_field = num_fields
        self.row, self.col = list(), list()
        for i in range(num_field - 1):
            for j in range(i + 1, num_field):
                self.row.append(i), self.col.append(j)
        torch.nn.init.xavier_uniform_(self.kernel.data)


    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        p, q = x[:, self.row], x[:, self.col]  # B,n,emb

        if self.kernel_type == 'mat':
            #  p [b,1,num_ix,e]
            #  kernel [e, num_ix, e]
            kp = torch.sum(p.unsqueeze(1) * self.kernel,dim=-1).permute(0,2,1)  #b,num_ix,e
            # #b,num_ix,e
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)