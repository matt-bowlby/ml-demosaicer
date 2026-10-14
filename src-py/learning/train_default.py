from torch.utils.data import DataLoader
import torch

from train_config import TrainConfig
from .model import Demosaicer
from .result import TrainResult
from .loss import reconstruction_loss


def train_default(
    data_loader: DataLoader, config: TrainConfig, model: Demosaicer | None = None
):
    # Initialize model and device target
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set manual seed for random values in torch if available
    if config.training_seed is not None:
        torch.manual_seed(config.training_seed)

    # If model is not explicitly given, create it; set device and training mode.
    if model is None:
        model = Demosaicer()
    model = model.to(device)
    model.train()

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    loss_function = reconstruction_loss

    # Create train result
    result = TrainResult()
    for _ in range(config.epoch_num):
        batch = 1
        for x, y, seed in data_loader:
            batch += 1

            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            # Clear last gradient; otherwise PyTorch accumulates gradients
            optimizer.zero_grad()
            # Make prediction
            prediction = model(x)
            # Calculate distance from correct answer
            loss = loss_function(prediction, y, 3.0)
            # Measure result
            with torch.no_grad():
                result.measure(x, prediction, y)
            # Calculate loss gradient
            loss.backward()
            # Step the optimizer toward the correct answer
            optimizer.step()

            # Yield progress
            yield model, result, config

    return model, result, config
