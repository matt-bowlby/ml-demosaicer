from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from config import TrainConfig
from .model import Demosaicer
from learning.dataset import RawDataset, SingleImageDataset

from debug import start, stop, reset


def train(config: TrainConfig):
    # Initialize dataset
    data_set = SingleImageDataset(config) if config.single_image else RawDataset(config)
    data_loader = DataLoader(
        data_set,
        config.batch_size,
    )

    # Initialize model and device target
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.training_seed is not None:
        torch.manual_seed(config.training_seed)
    model = Demosaicer().to(device)
    model.train()

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    loss_function = nn.L1Loss()

    result = TrainResult()

    reset()

    batch = 1
    start("Generate images for batch 1")
    for x, y, seed in data_loader:
        stop()
        start(f"Batch {batch}")
        batch += 1

        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        # Clear last gradient; otherwise PyTorch accumulates gradients
        optimizer.zero_grad()
        # Make prediction
        prediction = model(x)
        # Calculate distance from correct answer
        loss = loss_function(prediction, y)
        with torch.no_grad():
            result.measure(x, prediction, y)
        # Calculate loss gradient
        loss.backward()
        # Step the optimizer toward the correct answer
        optimizer.step()

        stop()
        config.update_run(seed, x.shape[0], None, config.image_generation_version)
        start(f"Generate Images for batch {batch}")

    return model, result, config

