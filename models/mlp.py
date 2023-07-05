import torch
from torch import nn
import torch.nn.functional as F


class mlp_factorized(nn.Module):
    def __init__(self, input_dim, num_hidden, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)

        self.linear3 = nn.Linear(num_hidden, num_hidden)
        self.linear4 = nn.Linear(num_hidden, num_hidden)

        self.linear5 = nn.Linear(num_hidden, num_hidden)
        self.fc = nn.Linear(num_hidden, num_classes)

        self.grouped_layers = [[self.linear1, self.linear2], [self.linear3, self.linear4], [self.linear5, self.fc]]
        self.other_layers = []

        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        x = F.relu(self.linear3(x))
        x = self.linear4(x)

        x = F.relu(self.linear5(x))
        out = self.fc(x)

        return out


class mlp(nn.Module):
    def __init__(self, input_dim, num_hidden, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, num_hidden)  # W1+b_w1
        self.linear2 = nn.Linear(num_hidden, num_hidden)  # V1+b_v1

        self.linear3 = nn.Linear(num_hidden, num_hidden)  # W2+b_w2
        self.linear4 = nn.Linear(num_hidden, num_hidden)  # V2+b_v2

        self.linear5 = nn.Linear(num_hidden, num_hidden)  # W3+b_w3
        self.fc = nn.Linear(num_hidden, num_classes)  # V3+b_v3

        self.grouped_layers = [[self.linear1, self.linear2], [self.linear3, self.linear4], [self.linear5, self.fc]]
        self.other_layers = []
        self.num_classes = num_classes

    def forward(self, x):
        x = torch.flatten(x, 1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        x = F.relu(self.linear5(x))
        out = self.fc(x)

        return out