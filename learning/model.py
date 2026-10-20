import torch.nn as nn

class Demosaicer(nn.Module):
    """Predict full-resolution RGB from sparse RGB Bayer input (N, 3, H, W)."""

    def __init__(self):
        super().__init__()
        self.activation = nn.LeakyReLU(0.2)

        self.feature_layer = nn.Conv2d(3, 64, 3, padding=1, padding_mode="reflect")
        self.inner_layer_1 = nn.Conv2d(64, 128, 3, padding=1, padding_mode="reflect")
        self.inner_layer_2 = nn.Conv2d(128, 64, 3, padding=1, padding_mode="reflect")
        self.output_layer = nn.Conv2d(64, 3, 3, padding=1, padding_mode="reflect")

    def forward(self, x):
        feature_layer_result = self.activation(self.feature_layer(x))
        inner_layer_1_result = self.activation(self.inner_layer_1(feature_layer_result))
        inner_layer_2_result = self.activation(self.inner_layer_2(inner_layer_1_result))
        output_layer_result = self.output_layer(inner_layer_2_result)
        combined = output_layer_result + x
        return combined
