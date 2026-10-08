import numpy as np
import torch

from generate_image import generate_image, bayer_mosaic
from config import TrainConfig
from learning.train import train
from learning.eval import predict, evaluate
from learning.model import Demosaicer
from learning.dataset import EvalDataset


def main():
    config = TrainConfig("config.json", "training_runs.json")

    training = False

    train_result = None
    if training:
        model, train_result, _ = train(config)
        torch.save(model.state_dict(), "model_sparse_rgb_32x32.pt")
    else:
        model = Demosaicer()
        weights = torch.load(
            "model_sparse_rgb_32x32.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(weights)

    if config.single_image:
        rng = np.random.default_rng(config.training_seed)
    else:
        rng = np.random.default_rng()
        seed = rng.integers(config.testing_range_start, config.testing_range_end)
        rng = np.random.default_rng(seed)
    # img = generate_image(rng, 32, 32, 1)
    # mosaic = bayer_mosaic(img, "RGGB")
    # predict(model, mosaic, img)

    eval_loader = EvalDataset(config)
    eval_result = evaluate(model, eval_loader, config)
    eval_result.add_train_result(train_result)
    eval_result.visualize()


if __name__ == "__main__":
    main()
