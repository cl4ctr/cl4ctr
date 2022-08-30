import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.FM import FactorizationMachineModel, FM_CL4CTR
from model.DeepFM import DeepFM, DeepFM_CL4CTR

import numpy as np
import random
import sys
import tqdm
import time
import argparse
import torch

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn.metrics import log_loss, roc_auc_score

sys.path.append("../..")
from dataloader.frappe.dataloader import getdataloader_ml, getdataloader_frappe

from utils.utils_de import *
from utils.earlystoping import EarlyStopping

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(
        name,
        field_dims,
        batch_size=1024,
        pratio=0.5,
        embed_dim=20,
        mlp_layers=(400, 400, 400)):
    if name == "fm_cl4ctr":
        return FM_CL4CTR(field_dims, embed_dim, batch_size=batch_size, pratio=pratio, fi_type="att")
    elif name == "dfm_cl4ctr":
        return DeepFM_CL4CTR(field_dims, embed_dim, mlp_layers=mlp_layers, batch_size=batch_size, pratio=pratio,
                             fi_type="att")
    else:
        raise ValueError('unknown model name: ' + name)


def count_params(model):
    params = sum(param.numel() for param in model.parameters())
    return params


def train(model,
          optimizer,
          data_loader,
          criterion,
          alpha=1.0,
          beta=1e-2):
    model.train()
    pred = list()
    target = list()
    total_loss = 0
    for i, (user_item, label) in enumerate(tqdm.tqdm(data_loader)):
        label = label.float()
        user_item = user_item.long()

        user_item = user_item.cuda()
        label = label.cuda()

        model.zero_grad()
        pred_y = torch.sigmoid(model(user_item).squeeze(1))
        loss_y = criterion(pred_y, label)

        # 1. Utilize simplified method to compute feature alignment and field uniformity
        loss = loss_y + model.compute_cl_loss(user_item, alpha=alpha, beta=beta)

        # 2. Utilize completely method to compute feature alignment and field uniformity
        # loss = loss_y + model.compute_cl_loss_all(user_item, alpha=alpha, beta=beta)

        loss.backward()
        optimizer.step()

        pred.extend(pred_y.tolist())
        target.extend(label.tolist())
        total_loss += loss.item()
        # if (i + 1) % log_interval == 0:
        #     print('train_loss:', total_loss / (i + 1))
        #     print(f'loss_y:{loss_y.item()};loss_cl:{loss_cl.item()}')
        # print("logloss",log_loss(target,pred))

    ave_loss = total_loss / (i + 1)
    return ave_loss


def test_roc(model, data_loader):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(
                data_loader, smoothing=0, mininterval=1.0):
            fields = fields.long()
            target = target.float()
            fields, target = fields.cuda(), target.cuda()
            y = torch.sigmoid(model(fields).squeeze(1))

            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def main(dataset_name, model_name, epoch, embed_dim, learning_rate,
         batch_size, weight_decay, save_dir, path,
         pratio, alpha, beta):
    path = "./data/"
    field_dims, trainLoader, validLoader, testLoader = \
        getdataloader_ml(path=path, batch_size=batch_size)
    print(field_dims)
    time_fix = time.strftime("%m%d%H%M%S", time.localtime())
    for K in [embed_dim]:
        paths = os.path.join(save_dir, dataset_name, model_name, str(K))
        if not os.path.exists(paths):
            os.makedirs(paths)
        with open(paths + f"/{model_name}_{K}_{batch_size}_{alpha}_{beta}_{pratio}_{time_fix}.p",
                  "a+") as fout:
            fout.write("Batch_size:{}\tembed_dim:{}\tlearning_rate:{}\tStartTime:{}\tweight_decay:{}\tpratio:{}\t"
                       "\talpha:{}\tbeta:{}\t\n"
                       .format(batch_size, K, learning_rate, time.strftime("%d%H%M%S", time.localtime()), weight_decay,
                               pratio, alpha, beta))
            print("Start train -- K : {}".format(K))

            criterion = torch.nn.BCELoss()
            model = get_model(
                name=model_name,
                field_dims=field_dims,
                batch_size=batch_size,
                embed_dim=K,
                pratio=pratio).cuda()

            params = count_params(model)
            fout.write("count_params:{}\n".format(params))
            print(params)

            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

            # Initial EarlyStopping
            early_stopping = EarlyStopping(patience=8, verbose=True, prefix=path)
            scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, patience=4)
            val_auc_best = 0
            auc_index_record = ""

            val_loss_best = 1000
            loss_index_record = ""

            for epoch_i in range(epoch):
                print(__file__, model_name, K, epoch_i, "/", epoch)
                print("Batch_size:{}\tembed_dim:{}\tlearning_rate:{}\tStartTime:{}\tweight_decay:{}\tpratio:{}\t"
                      "\talpha:{}\tbeta:{}\t"
                      .format(batch_size, K, learning_rate, time.strftime("%d%H%M%S", time.localtime()), weight_decay,
                              pratio, alpha, beta))
                start = time.time()

                train_loss = train(model, optimizer, trainLoader, criterion, alpha=alpha, beta=beta)
                val_auc, val_loss = test_roc(model, validLoader)
                test_auc, test_loss = test_roc(model, testLoader)

                scheduler.step(val_auc)
                end = time.time()
                if val_loss < val_loss_best:
                    # torch.save({"state_dict": model.state_dict(), "best_auc": val_auc_best},
                    #            paths + f"/{model_name}_final_{K}_{time_fix}.pt")
                    torch.save(model, paths + f"/{model_name}_best_auc_{K}_{pratio}_{time_fix}.pkl")

                if val_auc > val_auc_best:
                    val_auc_best = val_auc
                    auc_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                if val_loss < val_loss_best:
                    val_loss_best = val_loss
                    loss_index_record = "epoch_i:{}\t{:.6f}\t{:.6f}".format(epoch_i, test_auc, test_loss)

                print(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_loss, test_auc))

                fout.write(
                    "Train  K:{}\tEpoch:{}\ttrain_loss:{:.6f}\tval_loss:{:.6f}\tval_auc:{:.6f}\ttime:{:.6f}\ttest_loss:{:.6f}\ttest_auc:{:.6f}\n"
                    .format(K, epoch_i, train_loss, val_loss, val_auc, end - start, test_loss, test_auc))

                early_stopping(val_auc)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                  .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))

            fout.write("Test:{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                       .format(K, val_auc, val_auc_best, val_loss, val_loss_best, test_loss, test_auc))

            fout.write("auc_best:\t{}\nloss_best:\t{}".format(auc_index_record, loss_index_record))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    # CUDA_VISIBLE_DEVICES=1 python main_ml_base.py --choice  0

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ml_tag', help="")
    parser.add_argument('--save_dir', default='chkpt_ml_tag', help="")
    parser.add_argument('--path', default="../data/", help="")
    parser.add_argument('--model_name', default='fm', help="")
    parser.add_argument('--epoch', type=int, default=5, help="")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch_size")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="")
    parser.add_argument('--device', default='cuda:0', help="cuda:0")
    parser.add_argument('--choice', default=0, type=int, help="choice")
    parser.add_argument('--hint', default="CL4CTR", help="")
    parser.add_argument('--embed_dim', default=5, type=int, help="the size of feature dimension")
    parser.add_argument('--pratio', default=0.5, type=float, help="pratio")
    parser.add_argument('--alpha', default=1e-0, type=float, help="alpha")
    parser.add_argument('--beta', default=1e-2, type=float, help="beta")
    args = parser.parse_args()

    if args.choice == 0:
        model_names = ["fm_cl4ctr"] * 1

    elif args.choice == 1:
        model_names = ["dfm_cl4ctr"] * 1

    print(model_names)

    for name in model_names:
        seed = np.random.randint(0, 100000)
        setup_seed(seed)
        main(dataset_name=args.dataset_name,
             model_name=name,
             epoch=args.epoch,
             learning_rate=args.learning_rate,
             batch_size=args.batch_size,
             weight_decay=args.weight_decay,
             save_dir=args.save_dir,
             path=args.path,
             pratio=args.pratio,
             embed_dim=args.embed_dim,
             alpha=args.alpha,
             beta=args.beta
             )
