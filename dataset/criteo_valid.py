import math
import shutil
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import lmdb
import numpy as np
import torch.utils.data
from tqdm import tqdm
import random
import pickle
import os


class CriteoDataset(torch.utils.data.Dataset):
    """
    Criteo Display Advertising Challenge Dataset

    Data prepration:
        * Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature
        * Discretize numerical values by log2 transformation which is proposed by the winner of Criteo Competition

    :param dataset_path: criteo train.txt path.
    :param cache_path: lmdb cache path.
    :param rebuild_cache: If True, lmdb cache is refreshed.
    :param min_threshold: infrequent feature threshold.

    We first separate last 500K for testing. And the rest dataset for training and validation.

    Reference:
        https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
        https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
    """

    def __init__(self, dataset_path=None, cache_path='.criteo', rebuild_cache=False, min_threshold=10):
        self.NUM_FEATS = 39
        self.NUM_INT_FEATS = 13
        self.min_threshold = min_threshold
        self.prefix = "./data/criteo/"
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            # 如果没有缓存，还是构建缓存
            self.__build_cache(dataset_path, cache_path)
        #    通过从缓存中获取数据
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        # 必须要实现的 从缓存中读取数据
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        #     x, y
        return np_array[1:], np_array[0]

    def __len__(self):
        # 必须要实现的
        return self.length

    # 构建缓存
    def __build_cache(self, path, cache_path):
        temp_path = self.prefix + "train.txt"
        # count feature map
        feat_mapper, defaults = self.__get_feat_mapper(temp_path)
        # load feature map
        # feat_mapper, defaults = self.__read_train_all_feats()
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1

            # write field_dims
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())

            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __read_train_all_feats(self):
        return pickle.load(open(self.prefix+"train_all_feat.pkl", "rb"))

    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split('\t')

                if len(values) != self.NUM_FEATS + 1:
                    continue

                for i in range(1, self.NUM_INT_FEATS + 1):
                    feat_cnts[i][convert_numeric_feature(values[i])] += 1

                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1

        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}

        f = open("train_all_feat.pkl", "wb")
        pickle.dump((feat_mapper, defaults), f)

        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: setup lmdb')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[0])
                # 先数值型的变量
                for i in range(1, self.NUM_INT_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(convert_numeric_feature(values[i]), defaults[i])

                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer


@lru_cache(maxsize=None)
def convert_numeric_feature(val: str):
    if val == '':
        return 'NULL'
    v = int(val)
    if v > 2:
        return str(int(math.log(v) ** 2))
    else:
        return str(v)


def get_criteo_dataset_loader_valid(train_path="criteo_train_4000w.txt", test_path="criteo_test_500w.txt", batch_size=256):
    # the test_path maybe null, if it is, we need to split the train dataset
    print("start load dataset from valid ache")
    prefix = "/FRNet/data/criteo/"
    prefix_c = "/FRNet/"
    train_path = prefix + train_path
    test_path = prefix + test_path
    # We separate the original file to two file. In fact, we dont need to do it.
    # Torch can split the last 500K instances for testing.
    train_dataset = CriteoDataset(dataset_path=train_path, cache_path=prefix_c+".criteo_train_valid",rebuild_cache=False)
    field_dims = train_dataset.field_dims
    test_dataset = CriteoDataset(dataset_path=test_path, cache_path=prefix_c+".criteo_test_valid",rebuild_cache=False)
    train_length = len(train_dataset)

    # 500K的验证集合，训练集合也是500K
    valid_length = 5000000

    # 其实这里可以不使用验证集合
    # 考虑如何划分数据集
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_length - valid_length, valid_length])
    # train_dataset, valid_dataset = train_dataset[:(train_length - valid_length) + 1], train_dataset[-valid_length:]

    print("train_dataset length:", len(train_dataset))
    print("valid_dataset length:", len(valid_dataset))
    print("test_dataset length:", len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,drop_last=True,pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4,drop_last=False,pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,drop_last=False,pin_memory=True)

    # pickle.dump((train_dataset,test_dataset,field_dims),open("all_data.pkl","wb"))
    # 3500w 500w 500w
    return field_dims,train_loader,valid_loader,test_loader


"""
the full data: train.txt, which contains about 45,000,000 simple, 13 numerical features and 26 categorical feature
[193949, 192773, 189657, 181075, 138801, 59697, 45571, 42646, 11970, 11422, 
5178, 4654, 3175, 2032, 1458, 634, 555, 306, 223, 126, 
118, 101, 95, 86, 84, 76, 75, 49, 45, 40, 
30, 27, 19, 18, 16, 11, 9, 5, 4]
feature count: 1086810
"""
