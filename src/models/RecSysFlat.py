from torch import nn
import torch


class RecSysGarbageNet(nn.Module):
    def __init__(self, n_user, n_item):
        super(RecSysGarbageNet, self).__init__()
        self.alfa = nn.Parameter(torch.zeros([1,1]),requires_grad=True)
        self.beta_u = nn.Embedding(n_user, 1)
        self.beta_i = nn.Embedding(n_item, 1)

    def forward(self,x):
        x = self.beta_u(x[:,0]) + self.beta_i(x[:,1])
        out = x + self.alfa
        return out


class RecSysGarbageNetV2(nn.Module):
    def __init__(self, n_user, n_item, n_factor):
        super(RecSysGarbageNetV2, self).__init__()

        # rec matrices
        self.P = nn.Embedding(n_user, n_factor)
        self.Q = nn.Embedding(n_item, n_factor)

        # biases
        self.beta_u = nn.Embedding(n_user, 1)
        self.beta_i = nn.Embedding(n_item, 1)
        self.alfa = nn.Parameter(torch.zeros([1,1]),requires_grad=True)

    def forward(self,x):
        P_u = self.P(x[:,0]).squeeze()
        Q_i = self.Q(x[:,1]).squeeze()
        b_u = self.beta_u(x[:,0]).squeeze()
        b_i = self.beta_i(x[:,1]).squeeze()
        alfa = self.alfa
        outputs = (P_u * Q_i).sum(axis=1) + b_u + b_i + alfa
        return outputs.flatten()