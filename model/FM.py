#!/usr/bin/env python
# -*- coding:utf-8 -*-

from model.BasiclLayer import BasicCTR, BasicCL4CTR, FactorizationMachine


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


class FM_CL4CTR(BasicCL4CTR):
    # Extends BasicCL4CTR, which integrate contrastive learning approach for CTR models.
    def __init__(self, field_dims, embed_dim, batch_size=1024, pratio=0.5, fi_type="att"):
        super(FM_CL4CTR, self).__init__(field_dims, embed_dim, batch_size, pratio=pratio, fi_type=fi_type)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: B,F
        :return:
        """
        emb_x = self.embedding(x)
        x = self.lr(x) + self.fm(emb_x)
        return x
