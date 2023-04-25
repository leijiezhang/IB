import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth, kernel_size, padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels*depth, kernel_size=kernel_size, groups=in_channels,
                                   bias=bias, padding=padding)
        self.pointwise = nn.Conv2d(out_channels*depth, out_channels, kernel_size=1, bias=bias, padding=padding)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64,
                 dropoutRate=0.5, kernLength=64, n_time=200,
                 F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        self.T = 120
        self.dr_rate = dropoutRate
        # Layer 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, False)

        # Layer 2
        self.depth_wise_conv2 = nn.Conv2d(F1, F1*D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1*D, False)
        self.pooling2 = nn.AvgPool2d(1, 4)

        # Layer 3
        self.conv3 = SeparableConv2d(F1*D, F2, 1, (1, 16), padding="same")
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
        self.pooling3 = nn.AvgPool2d((1, 8))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(F2*(n_time//32), nb_classes)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x.unsqueeze(1))
        x = self.batchnorm1(x)
        x = F.dropout(x, self.dr_rate)

        # Layer 2
        x = F.elu(self.depth_wise_conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, self.dr_rate)
        x = self.pooling2(x)

        # Layer 3
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, self.dr_rate)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(x.shape[0], -1)
        # x = F.sigmoid(self.fc1(x))
        x = F.elu(self.fc1(x))
        return x


class EEGNetReg(nn.Module):
    def __init__(self, nb_classes, Chans=64,
                 dropoutRate=0.5, kernLength=64, n_time=200,
                 F1=8, D=2, F2=16):
        super(EEGNetReg, self).__init__()
        self.T = 120
        self.dr_rate = dropoutRate
        # Layer 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, False)

        # Layer 2
        self.depth_wise_conv2 = nn.Conv2d(F1, F1*D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1*D, False)
        self.pooling2 = nn.AvgPool2d(1, 4)

        # Layer 3
        self.conv3 = SeparableConv2d(F1*D, F2, 1, (1, 16), padding="same")
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
        self.pooling3 = nn.AvgPool2d((1, 8))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(F2*(n_time//32), nb_classes)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x.unsqueeze(1))
        x = self.batchnorm1(x)
        x = F.dropout(x, self.dr_rate)

        # Layer 2
        x = F.elu(self.depth_wise_conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, self.dr_rate)
        x = self.pooling2(x)

        # Layer 3
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, self.dr_rate)
        x = self.pooling3(x)

        # FC Layer
        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.fc1(x))
        return x