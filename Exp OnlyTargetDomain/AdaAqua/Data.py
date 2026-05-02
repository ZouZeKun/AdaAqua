import h5py

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader


def seq2instance_generator(data, sequence_len):
    _, num_step, _ = data.shape
    max_start = num_step - sequence_len + 1
    for j in range(max_start): 
        x = data[:, j:j+sequence_len, :-1]
        y = data[:, j:j+sequence_len, -1:]
        mean = data[:, j:j+sequence_len, -2:-1]
        y = (y - mean)
        yield (x, y)

class CustomDataset(IterableDataset):
    def __init__(self, args, dataset_type='train'):
        self.args = args
        self.dataset_type = dataset_type

        with h5py.File(args.target_data_path, 'r') as file:
            Target_data = file['demand_data'][:]

        Target_data = torch.FloatTensor(Target_data)

        train_steps_target = 96 * 28
        val_steps_target   = 96 * 14
        test_steps_target  = 96 * 8

        # 拆分源域为三个子集
        if dataset_type == 'train':
            target_data = Target_data[:, :train_steps_target, :]
        elif dataset_type == 'val':
            target_data = Target_data[:, train_steps_target : train_steps_target+val_steps_target, :]
        elif dataset_type == 'test':
            target_data = Target_data[:, train_steps_target+val_steps_target : train_steps_target+val_steps_target+test_steps_target, :]

        self.target_data = target_data

    def __iter__(self):
        # 每次迭代都创建新的生成器
        target_generator = seq2instance_generator(self.target_data, self.args.sequence_len)
        for x_t, y_t in target_generator:
            yield (
                x_t.to(self.args.device),
                y_t.to(self.args.device)
            )