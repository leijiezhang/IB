import torch
import numpy as np


def mixup_data(x, y, alpha=1.0, mixup_rate=0.1):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size_select = round(x.size()[0]*mixup_rate)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)[0:batch_size_select]

    mixed_x = lam * x[0:batch_size_select, :] + (1 - lam) * x[index, :]
    # y_a, y_b = y, y[index]
    mixed_y = lam * y[0:batch_size_select, :] + (1 - lam) * y[index, :]

    aug_x = torch.cat([x, mixed_x], 0)
    aug_y = torch.cat([y, torch.round(mixed_y)], 0)
    return aug_x, aug_y


def meanup_data(x, y, gap=5, mixup_rate=0.1):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    n_batch = x.shape[0]
    max_group_idx = torch.ceil(n_batch-gap+1)
    for i in range(max_group_idx):
        x = torch.cat([x, x[i:i+gap, :, :].mean(0)], 0)
        y = torch.cat([y, y[i:i + gap, :].mean(0)], 0)

    return x, y
