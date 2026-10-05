import torch.nn as nn


class Demosaicer(nn.Module):
    """Predict full-resolution RGB from sparse RGB Bayer input (N, 3, H, W)."""

    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Conv2d(3, 32, 3, padding=1, padding_mode="reflect")
        self.layer_2 = nn.LeakyReLU()
        self.layer_3 = nn.Conv2d(32, 64, 3, padding=1, padding_mode="reflect")
        self.layer_4 = nn.LeakyReLU()
        self.layer_5 = nn.Conv2d(64, 32, 3, padding=1, padding_mode="reflect")
        self.layer_6 = nn.LeakyReLU()
        self.layer_7 = nn.Conv2d(32, 3, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        result = self.layer_1(x)
        result = self.layer_2(result)
        result = self.layer_3(result)
        result = self.layer_4(result)
        result = self.layer_5(result)
        result = self.layer_6(result)
        return self.layer_7(result)
