import torch
import torch.nn as nn
import torch.nn.functional as F


class Guidance(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(Guidance, self).__init__()
        self.in_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.out_dim = output_dim
        # 第一层全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        # 第二层全连接层
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # 输出层
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x, t):
        t = t.view(t.size(0), 1)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, t), dim=1)
        if x.size(1) < self.in_dim:
            pad_size = self.in_dim - x.size(1)
            x = F.pad(x, (0, pad_size), "constant", 0)
        elif x.size(1) > self.in_dim:
            x = x[:, :self.in_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
