import h5py

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader

def split_source_into_three(data, interval=3):
    splits = []
    min_length = (data.shape[1] // interval) * interval  # 截断为 interval 的整数倍
    data = data[:, :min_length]
    for start in range(interval):
        splits.append(data[:, start::interval, :])
    return splits

def seq2instance_generator(data, sequence_len, domain_label):
    _, num_step, _ = data.shape
    max_start = num_step - sequence_len + 1
    for j in range(max_start): 
        x = data[:, j:j+sequence_len, :-1]
        y = data[:, j:j+sequence_len, -1:]
        mean = data[:, j:j+sequence_len, -2:-1]
        y = (y - mean)
        yield (x, y, domain_label)

class CustomDataset(IterableDataset):
    def __init__(self, args, dataset_type='train'):
        self.args = args
        self.dataset_type = dataset_type
        self.num_buildings_source = args.num_buildings_source

        with h5py.File(args.source_data_path, 'r') as file:
            Source_data = file['demand_data'][:]
        with h5py.File(args.target_data_path, 'r') as file:
            Target_data = file['demand_data'][:]

        Source_data = torch.FloatTensor(Source_data)
        Target_data = torch.FloatTensor(Target_data)

        train_steps_target = 96 * 28
        train_steps_source = 3 * train_steps_target
        val_steps_target   = 96 * 14
        val_steps_source   = 3 * val_steps_target
        test_steps_target  = 96 * 8
        test_steps_source  = 3 * test_steps_target

        # 拆分源域为三个子集
        if dataset_type == 'train':
            source_split = split_source_into_three(Source_data[:self.num_buildings_source, :train_steps_source, :])
            target_data = Target_data[:, :train_steps_target, :]
        elif dataset_type == 'val':
            source_split = split_source_into_three(Source_data[:self.num_buildings_source, train_steps_source : train_steps_source+val_steps_source, :])
            target_data = Target_data[:, train_steps_target : train_steps_target+val_steps_target, :]
        elif dataset_type == 'test':
            source_split = split_source_into_three(Source_data[:self.num_buildings_source, train_steps_source+val_steps_source : train_steps_source+val_steps_source+test_steps_source, :])
            target_data = Target_data[:, train_steps_target+val_steps_target : train_steps_target+val_steps_target+test_steps_target, :]

        self.source_splits = source_split
        self.target_data = target_data
        
        # 源域生成器(三个子集)
        self.source_domain_labels = [torch.full((args.num_buildings_source,), -1.0)] * len(self.source_splits)
        # 目标域生成器(单个集合)
        if dataset_type == 'train':
            self.target_domain_label = torch.ones(args.num_buildings_target)
        elif dataset_type == 'val':
            self.target_domain_label = torch.ones(args.num_buildings_target)
        elif dataset_type == 'test':
            self.target_domain_label = torch.ones(args.num_buildings_target)

    def __iter__(self):
        # 每次迭代都会创建新的迭代器
        source_generators = [
            seq2instance_generator(split, self.args.sequence_len, label)
            for split, label in zip(self.source_splits, self.source_domain_labels)
        ]
        target_generator = seq2instance_generator(self.target_data, self.args.sequence_len, self.target_domain_label)

        # 创建 (多个源域-单个目标域) 的迭代器
        source_iterator = [iter(gen) for gen in source_generators]
        target_iterator = iter(target_generator)

        while True:
            available = []

            for src_its in source_iterator:
                try:
                    x_s, y_s, d_s = next(src_its)
                    available.append((x_s, y_s, d_s))
                except StopIteration:
                    pass

            if not available:
                break  # 所有源生成器都已完成

            try:
                x_t, y_t, d_t = next(target_iterator)
            except StopIteration:
                # 迭代完毕，重新迭代目标域
                target_iterator = iter(target_generator)
                x_t, y_t, d_t = next(target_iterator)

            for x_s, y_s, d_s in available:
                yield (
                    x_s.to(self.args.device),
                    y_s.to(self.args.device),
                    d_s.to(self.args.device),
                    x_t.to(self.args.device),
                    y_t.to(self.args.device),
                    d_t.to(self.args.device)
                )