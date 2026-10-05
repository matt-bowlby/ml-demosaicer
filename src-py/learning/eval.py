import numpy as np
import torch
import matplotlib.pyplot as plt

from generate_image.generate import to_HWC

def predict(model, bayer_mosaic, answer, display: bool = True) -> np.ndarray:
    model.eval()
    parameter = next(model.parameters())
    mosaic = torch.as_tensor(
        bayer_mosaic, dtype=parameter.dtype, device=parameter.device
    ).unsqueeze(0)

    with torch.no_grad():
        img = to_HWC(model(mosaic).squeeze(0).cpu().numpy())

        if display:
            fig, axes = plt.subplots(2, 2)
            fig.suptitle("Model output comparison")
            axes = axes.flatten()

            axes[0].imshow(to_HWC(bayer_mosaic))
            axes[0].set_title("Bayer Mosaic")

            axes[1].imshow(to_HWC(answer))
            axes[1].set_title("Answer")

            axes[2].imshow(img)
            axes[2].set_title("Model's guess")

            compare_img = compare(img, to_HWC(answer))
            axes[3].imshow(compare_img)
            axes[3].set_title("Comparison")

            for ax in axes:
                ax.axis("off")

            plt.tight_layout()
            plt.show()

        return img

def compare(a, b) -> np.ndarray:
    error_vis = np.abs(a - b) * 10
    return np.clip(error_vis, 0, 1)