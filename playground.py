import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, param):
        super(Unet, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.device_count() > 1 else 'cpu')
        self.param = param

    def build(self):
        l = self.param['len']

        feature_size = 8
        self.lstm1 = nn.LSTM(l, hidden_size=feature_size)

        feature_size *= 2
        self.lstm2 = nn.LSTM(l, hidden_size=feature_size)

        feature_size *= 2
        self.lstm3 = nn.LSTM(l, hidden_size=feature_size)

        self.att = nn.AdaptiveAvgPool1d(feature_size)
        self.att_fc = nn.Linear(feature_size)


    def forward(self, x):
        x = self.lstm1(x)
        x = F.relu(x)
        x = self.lstm2(x)
        x = F.relu(x)
        x = self.lstm3(x)
        x = F.relu(x)
        a = self.att(x)
        a = self.att_fc(a)

        x = x*a

        return x





