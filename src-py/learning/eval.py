import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from generate_image.generate import to_HWC

from .result import EvalResult


def predict(model, mosaic: torch.Tensor, answer: torch.Tensor, display: bool = True) -> torch.Tensor:
    model.eval()

    with torch.no_grad():
        prediction_img = to_HWC(model(mosaic.unsqueeze(0)).squeeze(0))
        mosaic_img = to_HWC(mosaic)
        answer_img = to_HWC(answer)
        compare_img = compare(prediction_img, answer_img).detach().cpu()
        prediction_img = prediction_img.detach().cpu()
        mosaic_img = mosaic.detach().cpu()
        answer_img = answer_img.detach().cpu()

        if display:
            fig, axes = plt.subplots(2, 2)
            fig.suptitle("Model output comparison")
            axes = axes.flatten()

            axes[0].imshow(mosaic_img, interpolation="nearest")
            axes[0].set_title("Bayer Mosaic")

            axes[1].imshow(answer_img, interpolation="nearest")
            axes[1].set_title("Answer")

            axes[2].imshow(prediction_img, interpolation="nearest")
            axes[2].set_title("Model's guess")

            axes[3].imshow(compare_img, interpolation="nearest")
            axes[3].set_title("Difference * 10")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.show()

        return prediction_img


def compare(a, b) -> torch.Tensor:
    error_vis = torch.abs(a - b) * 10
    return torch.clip(error_vis, 0, 1)


def evaluate(model, eval_loader: DataLoader):
    was_training = model.training
    model.eval()

    result = EvalResult()

    try:
        with torch.inference_mode():
            for x, y, seed in eval_loader:
                prediction = model(x)
                result.measure(x, prediction, y, seed)
    finally:
        model.train(was_training)

    return result