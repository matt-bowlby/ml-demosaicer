import torch
import matplotlib.pyplot as plt
import numpy as np

from generate_image.generate import to_HWC


class TrainResult:
    def __init__(self):
        self._loss = []

    def measure(self, input, prediction, answer):
        self._loss.append((prediction - answer).abs().mean().item())

    def get_loss(self):
        return self._loss


class EvalResult:
    def __init__(self, num_examples: int = 4) -> None:
        if num_examples < 0:
            raise ValueError("num_worst_images must be nonnegative.")
        self._num_examples = num_examples
        self._loss = []
        self._psnr = []
        self._worst_predictions = []
        self._train_result = None
        self._mae_rgb = []

    def add_train_result(self, result: TrainResult | None):
        self._train_result = result

    def measure(self, input, prediction, answer):
        loss = (prediction - answer).abs().mean().item()
        self._loss.append(loss)

        mse = (prediction - answer).square().mean(dim=(1, 2, 3))
        psnr = (-10 * torch.log10(mse.clamp_min(1e-12))).detach().cpu()
        self._psnr.append(psnr)

        mae_rgb = (prediction - answer).abs().mean(dim=(2, 3))
        self._mae_rgb.append(mae_rgb.detach().cpu())

        for i, score in enumerate(psnr.tolist()):
            if self._num_examples == 0:
                break
            if (
                len(self._worst_predictions) < self._num_examples
                or score < self._worst_predictions[-1]["psnr"]
            ):
                self._worst_predictions.append(
                    {
                        "psnr": score,
                        "input": input[i].detach().cpu().numpy(),
                        "prediction": prediction[i].detach().cpu().numpy(),
                        "target": answer[i].detach().cpu().numpy(),
                    }
                )
                self._worst_predictions.sort(key=lambda image: image["psnr"])
                del self._worst_predictions[self._num_examples :]

    def worst_predictions(self) -> list:
        return list(self._worst_predictions)

    def mean_psnr(self) -> float:
        psnr = torch.cat(self._psnr)
        return psnr.mean().item()

    def median_psnr(self) -> float:
        psnr = torch.cat(self._psnr)
        return torch.quantile(psnr, 0.5).item()

    def p10_psnr(self) -> float:
        psnr = torch.cat(self._psnr)
        return torch.quantile(psnr, 0.1).item()

    def min_psnr(self) -> float:
        return torch.cat(self._psnr).min().item()

    def max_psnr(self) -> float:
        return torch.cat(self._psnr).max().item()

    def mean_mae_rgb(self):
        return torch.cat(self._mae_rgb, dim=0).mean(dim=0)

    def visualize(self):
        fig = plt.figure(figsize=(2000, 1000, "px"))
        subfigs = fig.subfigures(2, 2, squeeze=False)
        top_left = subfigs[0][0]
        top_right = subfigs[0][1]
        top_left.suptitle("Worst Images")
        bottom_left = subfigs[1][0]
        bottom_right = subfigs[1][1]

        worst_prediction_plot = top_left.subplots(self._num_examples, 4)

        for i, pred in enumerate(self._worst_predictions):
            input = to_HWC(pred.get("input"))
            worst_prediction_plot[i][0].imshow(input)
            worst_prediction_plot[i][0].set_axis_off()

            prediction = to_HWC(pred.get("prediction"))
            worst_prediction_plot[i][1].imshow(prediction)
            worst_prediction_plot[i][1].set_axis_off()

            target = to_HWC(pred.get("target"))
            worst_prediction_plot[i][2].imshow(target)
            worst_prediction_plot[i][2].set_axis_off()

            difference = np.abs(prediction - target) * 10
            worst_prediction_plot[i][3].imshow(difference)
            worst_prediction_plot[i][3].set_axis_off()

        loss_stats = top_right.subplots(2, 3)

        values = [
            (self.mean_mae_rgb()[0], "Red MAE"),
            (self.mean_mae_rgb()[1], "Green MAE"),
            (self.mean_mae_rgb()[2], "Blue MAE"),
        ]
        for ax, (value, title) in zip(loss_stats.flatten(), values):
            ax.text(
                0.5,
                0.5,
                f"{value:.4f}",
                fontsize=20,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            ax.set_title(title)

        psnr_stats = bottom_left.subplots(2, 3)

        x = np.arange(len(self._loss))
        y = self._loss
        psnr_stats[0][0].plot(x, y)
        psnr_stats[0][0].set_xlabel("Item")
        psnr_stats[0][0].set_ylabel("Loss")
        psnr_stats[0][0].set_title("Loss Over Eval Set")

        values = [
            (self.mean_psnr(), "Mean PSNR"),
            (self.p10_psnr(), "10th Percentile PSNR"),
            (self.min_psnr(), "Minimum PSNR"),
            (self.median_psnr(), "Median PSNR"),
            (self.max_psnr(), "Maximum PSNR"),
        ]
        for ax, (value, title) in zip(psnr_stats.flatten()[1:], values):
            ax.text(
                0.5,
                0.5,
                f"{value:.4f} db",
                fontsize=20,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            ax.set_title(title)

        if self._train_result is not None:
            y = self._train_result.get_loss()
            x = np.arange(len(y))
            ax = bottom_right.subplots()
            ax.plot(x, y)
            ax.set_xlabel("Item")
            ax.set_ylabel("Loss")
            ax.set_title("Loss Over Training Set")

        plt.show()
