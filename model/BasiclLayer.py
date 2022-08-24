import torch.nn as nn
import numpy as np
from .data_aug import *


class BasicCTR(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(BasicCTR, self).__init__()
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

    def forward(self, x):
        raise NotImplemented


class BasicCL4CTR(nn.Module):
    """
    The core implement of CL4CTR, in which three SSL losses(L_cl, L_ali and L_uni) are computed to regularize
    feature representation.
    """

    def __init__(self, field_dims, embed_dim, batch_size=1024, pratio=0.5, fi_type="att"):
        super(BasicCL4CTR, self).__init__()
        # 1、embedding layer
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.field_dims = field_dims
        self.num_field = len(field_dims)
        self.input_dim = self.num_field * embed_dim
        self.batch_size = batch_size
        self.row, self.col = list(), list()
        for i in range(batch_size - 1):
            for j in range(i + 1, batch_size):
                self.row.append(i), self.col.append(j)

        # 2.1 Random mask.
        self.pratio = pratio
        self.dp1 = nn.Dropout(p=pratio)
        self.dp2 = nn.Dropout(p=pratio)

        # 2.2 FI_encoder. In most cases, we utilize three layer transformer layers.
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=1, dim_feedforward=128,
                                                        dropout=0.2)
        self.fi_cl = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        # 2.3 Projection
        self.projector1 = nn.Linear(self.input_dim, embed_dim)
        self.projector2 = nn.Linear(self.input_dim, embed_dim)

    def forward(self, x):
        raise NotImplemented

    def compute_cl_loss(self, x, alpha=1.0, beta=0.01):
        """
        :param x: embedding
        :param alpha:
        :param beta: beta = gamma
        :return: L_cl * alpha + (L_ali+L_uni) * beta

        # This is a simplified computation based only on the embedding of each batch,
        # which can accelerate the training process.
        """
        x_emb = self.embedding(x)

        # 1. Compute feature alignment loss (L_ali) and feature uniformity loss (L_uni).
        # This is a simplified computation based only on the embedding of each batch,
        # which can accelerate the training process.
        cl_align_loss = self.compute_alignment_loss(x_emb)
        cl_uniform_loss = self.compute_uniformity_loss(x_emb)
        if alpha == 0.0:
            return (cl_align_loss + cl_uniform_loss) * beta

        # 2. Compute contrastive loss.

        x_emb1, x_emb2 = self.dp1(x_emb), self.dp2(x_emb)
        x_h1 = self.fi_cl(x_emb1).view(-1, self.input_dim)
        x_h2 = self.fi_cl(x_emb2).view(-1, self.input_dim)

        x_h1 = self.projector1(x_h1)
        x_h2 = self.projector2(x_h2)

        cl_loss = torch.norm(x_h1.sub(x_h2), dim=1).pow_(2).mean()

        # 3. Combine L_cl and (L_ali + L_uni) with two loss weights (alpha and beta)
        loss = cl_loss * alpha + (cl_align_loss + cl_uniform_loss) * beta
        return loss

    def compute_cl_loss_all(self, x, alpha=1.0, beta=0.01):
        """
        :param x: embedding
        :param alpha:
        :param beta: beta
        :return: L_cl * alpha + (L_ali+L_uni) * beta

        This is the full version of Cl4CTR, which computes L_ali and L_uni with full feature representations.
        """
        x_emb = self.embedding(x)

        # 1. Compute feature alignment loss (L_ali) and feature uniformity loss (L_uni).
        cl_align_loss = self.compute_all_alignment_loss()
        cl_uniform_loss = self.compute_all_uniformity_loss()
        if alpha == 0.0:
            return (cl_align_loss + cl_uniform_loss) * beta

        # 2. Compute contrastive loss (L_cl).
        x_emb1, x_emb2 = self.dp1(x_emb), self.dp2(x_emb)
        x_h1 = self.fi_cl(x_emb1).view(-1, self.input_dim)
        x_h2 = self.fi_cl(x_emb2).view(-1, self.input_dim)

        x_h1 = self.projector1(x_h1)
        x_h2 = self.projector2(x_h2)

        cl_loss = torch.norm(x_h1.sub(x_h2), dim=1).pow_(2).mean()

        # 3. Combine L_cl and (L_ali + L_uni) with two loss weights (alpha and beta)
        loss = cl_loss * alpha + (cl_align_loss + cl_uniform_loss) * beta
        return loss

    def compute_alignment_loss(self, x_emb):
        alignment_loss = torch.norm(x_emb[self.row].sub(x_emb[self.col]), dim=2).pow(2).mean()
        return alignment_loss

    def compute_uniformity_loss(self, x_emb):
        frac = torch.matmul(x_emb, x_emb.transpose(2, 1))  # B,F,F
        denom = torch.matmul(torch.norm(x_emb, dim=2).unsqueeze(2), torch.norm(x_emb, dim=2).unsqueeze(1))  # 64，30,30
        res = torch.div(frac, denom + 1e-4)
        uniformity_loss = res.mean()
        return uniformity_loss

    def compute_all_uniformity_loss(self):
        """
            Calculate field uniformity loss based on all feature representation.
        """
        embedds = self.embedding.embedding.weight
        field_dims = self.field_dims
        field_dims_cum = np.array((0, *np.cumsum(field_dims)))
        field_len = embedds.size()[0]
        field_index = np.array(range(field_len))
        uniformity_loss = 0.0
        #     for i in
        pairs = 0
        for i, (start, end) in enumerate(zip(field_dims_cum[:-1], field_dims_cum[1:])):
            index_f = np.logical_and(field_index >= start, field_index < end)  # 前闭后开
            embed_f = embedds[index_f, :]
            embed_not_f = embedds[~index_f, :]
            frac = torch.matmul(embed_f, embed_not_f.transpose(1, 0))  # f1,f2
            denom = torch.matmul(torch.norm(embed_f, dim=1).unsqueeze(1),
                                 torch.norm(embed_not_f, dim=1).unsqueeze(0))  # f1,f2
            res = torch.div(frac, denom + 1e-4)
            uniformity_loss += res.sum()
            pairs += (field_len - field_dims[i]) * field_dims[i]
        uniformity_loss /= pairs
        return uniformity_loss

    def compute_all_alignment_loss(self):
        """
        Calculate feature alignment loss based on all feature representation.
        """
        embedds = self.embedding.embedding.weight
        field_dims = self.field_dims
        field_dims_cum = np.array((0, *np.cumsum(field_dims)))
        alignment_loss = 0.0
        pairs = 0
        for i, (start, end) in enumerate(zip(field_dims_cum[:-1], field_dims_cum[1:])):
            embed_f = embedds[start:end, :]
            loss_f = 0.0
            for j in range(field_dims[i]):
                loss_f += torch.norm(embed_f[j, :].sub(embed_f), dim=1).pow(2).sum()
            pairs += field_dims[i] * field_dims[i]
            alignment_loss += loss_f

        alignment_loss /= pairs
        return alignment_loss


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
        square_of_sum = torch.sum(x, dim=1) ** 2  # B，embed_dim
        sum_of_square = torch.sum(x ** 2, dim=1)  # B，embed_dim
        ix = square_of_sum - sum_of_square  # B,embed_dim
        if self.reduce_sum:
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
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.mlp(x)
