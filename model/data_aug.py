import torch


def maskrandom(x_emb, mask_ratio):
    B, F, E = x_emb.size()
    mask1 = torch.bernoulli(torch.ones(B, F, E) * mask_ratio).cuda()
    mask2 = torch.bernoulli(torch.ones(B, F, E) * mask_ratio).cuda()
    x_emb1 = x_emb * mask1
    x_emb2 = x_emb * mask2
    return x_emb1, x_emb2


def maskdimension(x_emb, mask_ratio):
    B, F, E = x_emb.size()
    mask1 = torch.bernoulli(torch.ones(B, 1, E) * mask_ratio).cuda()
    mask2 = torch.bernoulli(torch.ones(B, 1, E) * mask_ratio).cuda()
    x_emb1 = x_emb * mask1
    x_emb2 = x_emb * mask2
    return x_emb1, x_emb2


def maskfeature(x_emb, mask_ratio):
    B, F, E = x_emb.size()
    mask1 = torch.bernoulli(torch.ones(B, F, 1) * mask_ratio).cuda()
    mask2 = torch.bernoulli(torch.ones(B, F, 1) * mask_ratio).cuda()
    x_emb1 = x_emb * mask1
    x_emb2 = x_emb * mask2
    return x_emb1, x_emb2
