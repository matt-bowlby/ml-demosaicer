import numpy as np
import matplotlib.pyplot as plt
import torch

from generate_image import generate_image, bayer_mosaic
from config import TrainConfig
from learning.train import train
from learning.eval import predict
from learning.model import Demosaicer

def main():
    config = TrainConfig("config.json", "training_runs.json")

    training = False

    if training:
        model, config = train(config)
        torch.save(model.state_dict(), "model_sparse_rgb_32x32.pt")
    else:
        model = Demosaicer()
        weights = torch.load(
            "model_sparse_rgb_32x32.pt",
            map_location="cpu",
            weights_only=True
        )
        model.load_state_dict(weights)

    rng = np.random.default_rng()
    rng = np.random.default_rng(rng.integers(config.testing_range_start, config.testing_range_end))
    img = generate_image(rng, 32, 32, 1)
    mosaic = bayer_mosaic(img, "RGGB")
    predict(model, mosaic, img)


if __name__ == "__main__":
    main()
