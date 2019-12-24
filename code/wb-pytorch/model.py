import torch
from torch import nn
from config import *


class BiLSTM(nn.Module):
    """
    BiLSTM + FC
    """
    def __init__(self, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.num_layers = 1
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(n_input, n_hidden, num_layers=self.num_layers, batch_first=False, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(in_features=n_hidden * 2, out_features=n_classes))

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        outputs, _ = self.lstm(x)
        # outputs = outputs.view(1, -1)
        outputs = self.dropout(outputs)
        # print(x.shape)
        # print(outputs.shape)
        val = self.fc(outputs[-1]).view(1, n_classes)
        return val
