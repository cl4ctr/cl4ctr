#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.FM import FactorizationMachineModel, FMFRNet
from model.DeepFM import DeepFM, DeepFM_FRNet

import numpy as np
import random

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.metrics import log_loss, roc_auc_score

sys.path.append("../..")
from dataset.frappe.dataloader import getdataloader_frappe, getdataloader_ml

from utils.utils_de import *
from utils.earlystoping import EarlyStopping,EarlyStoppingLoss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(
        name,
        field_dims,
        embed_dim=20,
        mlp_layers=(400, 400, 400)):

    if name == "fm":
        return FactorizationMachineModel(field_dims, embed_dim)
    elif name == "fmfrnet":
        return FMFRNet(field_dims,embed_dim,num_layers=2, weight_type="bit", att_size=20, mlp_layer=256)
    elif name == "fmfrnet_vec":
        return FMFRNet(field_dims,embed_dim,num_layers=2,weight_type="vector",att_size=20,mlp_layer=256)

    elif name == "dfm":
        return FactorizationMachineModel(field_dims, embed_dim)

    elif name == "dfmfrnet":
        return FMFRNet(field_dims, embed_dim, num_layers=2, weight_type="bit", att_size=20, mlp_layer=256)

    else:
        raise ValueError('unknown model name: ' + name)

def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params

def train(model,
          optimizer,
          data_loader,
          criterion):
    model.train()
    pred = list()
    target = list()
    total_loss = 0
    for i, (user_item, label) in enumerate(tqdm.tqdm(data_loader, ncols=80, position=0)):
        label = label.float()
        user_item = user_item.long()
        user_item = user_item.to(DEVICE)  # [B,F]
        label = label.to(DEVICE)   # [B]

        model.zero_grad()
        pred_y = torch.sigmoid(model(user_item).squeeze(1))
        loss = criterion(pred_y, label)
        loss.backward()
        optimizer.step()

        pred.extend(pred_y.tolist())
        target.extend(label.tolist())
        total_loss += loss.item()

    loss2 = total_loss / (i + 1)
    return loss2


def test_roc(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    for fields, target in tqdm.tqdm(
            data_loader, ncols=80, position=0):
        fields = fields.long()
        target = target.float()
        fields, target = fields.to(DEVICE), target.to(DEVICE)
        # add torch.sigmoid() for CTR prediction task.
        y = torch.sigmoid(model(fields).squeeze(1))

        targets.extend(target.tolist())
        predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def main(dataset_name,
          model_name,
          epoch,
          learning_rate,
          batch_size,
          weight_decay,
          save_dir,
          path,
          embed_dim,
          hint=""):
    path = "./data/"
    field_dims, trainLoader, validLoader, testLoader = \
        getdataloader_frappe(path=path, batch_size=batch_size)
    print(field_dims)
    print(sum(field_dims))
    time_fix = time.strftime("%d%H%M%S", time.localtime())

    for K in [embed_dim]:
        paths = os.path.join(save_dir, dataset_name, model_name, str(K))
        if not os.path.exists(paths):
            os.makedirs(paths)
        with open(paths + f"/{model_name}logs2_{K}_{batch_size}_{learning_rate}_{weight_decay}_{time_fix}.p", "a+") as fout:
            # 记录配置
            fout.write("Batch_size:{}\tlearning_rate:{}\tStartTime:{}\tweight_decay:{}\n"
                       .format(batch_size, learning_rate, time.strftime("%d%H%M%S", time.localtime()), weight_decay))
            print("Start train -- K : {}".format(K))
            criterion = torch.nn.BCELoss() # CTR
            model = get_model(
                name=model_name,
                field_dims=field_dims,
                embed_dim=K).to(DEVICE)

            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

            # EarlyStopping
            # early_stopping = EarlyStopping(patience=8, verbose=True, prefix=path)
            early_stopping = EarlyStoppingLoss(patience=8, verbose=True, prefix=path)

            val_auc_best = 0
            auc_index_record = ""

            val_loss_best = 1000
            loss_index_record = ""

            scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=4)
            for epoch_i in range(epoch):
                print(__file__, model_name, K, learning_rate, weight_decay, epoch_i,"/",epoch)

                start = time.time()

                train_loss = train(model, optimizer, trainLoader, criterion)
                val_auc, val_loss = test_roc(model, validLoader)
                test_auc, test_loss = test_roc(model, testLoader)
                scheduler.step(val_auc)
                end = time.time()

                if val_auc > val_auc_best:
                    val_auc_best = val_auc
                    auc_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)
                    # save model
                    torch.save(model, paths+f"/{model_name}_best_auc_{K}_{time_fix}.pkl")

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    loss_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                print(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                        .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_loss, test_auc))
                fout.write(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                        .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_loss, test_auc))

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                  .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))

            fout.write("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                       .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))
            fout.write("auc_best:\t{}\nloss_best:\t{}".format(auc_index_record, loss_index_record))
            # torch.save({"state_dict": model.state_dict()}, paths + f"/{model_name}_final_{K}_{time_fix}.pt")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    seed = np.random.randint(0, 1000)
    setup_seed(seed)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='chkpt0523')
    parser.add_argument('--dataset_name', default='frappe2')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cuda:0', help="cuda:0")
    parser.add_argument('--hint', default="FRNet")
    parser.add_argument('--path', default="./data/")
    parser.add_argument('--choice', default=0, type=int)
    parser.add_argument('--embed_dim', default=20, type=int)

    args = parser.parse_args()
    if args.choice == 0:
        model_names = ["dfmfrnet"] * 4
    elif args.choice == 1:
        model_names = ["fmfrnet"] * 2


    args.dataset_name = "frappe"
    # args.dataset_name = "frappe/gate2"
    print(model_names)
    for lr in [0.01]:
        for weight_decay in [1e-4]:
            args.learning_rate = lr
            args.weight_decay = weight_decay
            for name in model_names:
                main(dataset_name=args.dataset_name,
                      model_name=name,
                      epoch=args.epoch,
                      learning_rate=args.learning_rate,
                      batch_size=args.batch_size,
                      weight_decay=args.weight_decay,
                      save_dir=args.save_dir,
                      path=args.path,
                      embed_dim= args.embed_dim,
                      hint=args.hint)


