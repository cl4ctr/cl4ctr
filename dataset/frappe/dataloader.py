#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Author:wangfy
@project:DL_recommend
@Time:2020/4/29 9:42 上午
'''

import numpy as np
import pandas as pd
import torch
import os
import tqdm
import pickle


class LoadData():
    # 加载数据,
    def __init__(self, path="./data/", dataset="frappe"):
        self.dataset = dataset
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset + ".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        self.features_M = {}
        self.construct_df()

    #         self.Train_data, self.Validation_data, self.Test_data = self.construct_data( loss_type )

    def construct_df(self):
        self.data_train = pd.read_table(self.trainfile, sep=" ", header=None, engine='python')
        self.data_test = pd.read_table(self.testfile, sep=" ", header=None, engine="python")
        self.data_valid = pd.read_table(self.validationfile, sep=" ", header=None, engine="python")
        #       第一列是标签，y

        for i in self.data_test.columns[1:]:
            self.data_test[i] = self.data_test[i].apply(lambda x: int(x.split(":")[0]))
            self.data_train[i] = self.data_train[i].apply(lambda x: int(x.split(":")[0]))
            self.data_valid[i] = self.data_valid[i].apply(lambda x: int(x.split(":")[0]))

        self.all_data = pd.concat([self.data_train, self.data_test, self.data_valid])
        self.field_dims = []

        for i in self.all_data.columns[1:]:
            # if self.dataset != "frappe":
                # maps = {}
            maps = {val: k for k, val in enumerate(set(self.all_data[i]))}
            self.data_test[i] = self.data_test[i].map(maps)
            self.data_train[i] = self.data_train[i].map(maps)
            self.data_valid[i] = self.data_valid[i].map(maps)
            self.features_M[i] = maps
            self.field_dims.append(len(set(self.all_data[i])))
        # -1 改成 0
        self.data_test[0] = self.data_test[0].apply(lambda x: max(x, 0))
        self.data_train[0] = self.data_train[0].apply(lambda x: max(x, 0))
        self.data_valid[0] = self.data_valid[0].apply(lambda x: max(x, 0))


class RecData():
    # define the dataset
    def __init__(self, all_data):
        self.data_df = all_data

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        x = self.data_df.iloc[idx].values[1:]
        y1 = self.data_df.iloc[idx].values[0]
        return x, y1

def getdataloader_frappe(path="../.././data/",dataset="frappe",num_ng=4, batch_size=256):
    print("Load frappe dataset.")
    DataF = LoadData(path=path, dataset=dataset)

    datatest = RecData(DataF.data_test)
    datatrain = RecData(DataF.data_train)
    datavalid = RecData(DataF.data_valid)
    print("datatrain",len(datatrain))
    print("datavalid",len(datavalid))
    print("datatest",len(datatest))
    trainLoader = torch.utils.data.DataLoader(datatrain, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True)
    validLoader = torch.utils.data.DataLoader(datavalid, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)
    testLoader = torch.utils.data.DataLoader(datatest, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)
    return DataF.field_dims,trainLoader,validLoader,testLoader

def getdataloader_ml(path="../.././data/", dataset="ml-tag",num_ng=4, batch_size=256):

    path_ml = path + 'preprocess-ml.p'
    if not os.path.exists(path_ml):
        DataF = LoadData(path=path, dataset=dataset)
        # save data save time
        pickle.dump((DataF.data_test,DataF.data_train,DataF.data_valid,DataF.field_dims), open(path_ml, 'wb'))
        print("success")
    print("start load ml_tag data")
    data_test,data_train,data_valid, field_dims = pickle.load(open(path_ml, mode='rb'))
    datatest = RecData(data_test)
    datatrain = RecData(data_train)
    datavalid = RecData(data_valid)
    print("ml-datatrain",len(datatrain))
    print("ml-datavalid",len(datavalid))
    print("ml-datatest",len(datatest))
    trainLoader = torch.utils.data.DataLoader(datatrain, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True)
    validLoader = torch.utils.data.DataLoader(datavalid, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)
    testLoader = torch.utils.data.DataLoader(datatest, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)
    return field_dims,trainLoader,validLoader,testLoader

if __name__ == '__main__':
    field_dims,trainLoader,validLoader,testLoader = getdataloader_ml(batch_size=256)
    for _ in tqdm.tqdm(trainLoader):
        pass
    it = iter(trainLoader)
    print(next(it)[0])
    print(field_dims)


