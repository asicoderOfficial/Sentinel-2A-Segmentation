import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self, input_channels: int):
        super(Baseline, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        return self.model(x)
