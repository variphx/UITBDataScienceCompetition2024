import torch.nn as nn


class VimmsdModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = nn.LazyLinear(1024)

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(1024, 4),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.linear = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3,
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.tanh(x)
        return x
