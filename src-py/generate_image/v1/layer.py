import math
import torch
from .random import RandomStreams, rngr_control


def layer(background, shape, fill, opacity: torch.Tensor, *, rng: RandomStreams | None = None,
          center: tuple[float, float] | None = None) -> torch.Tensor:
    """Blend a local shape in place, using its bounding-box center as origin.

    Supply RandomStreams for random placement, or an explicit (x, y) center.
    Shapes extending past the image edges are clipped during compositing.
    """
    height, width = background.shape[:2]
    box_height, box_width = shape.shape[:2]
    if fill.shape != (box_height, box_width, 3):
        raise ValueError("fill must match the shape's bounding box and have three channels")
    if center is None:
        if rng is None:
            raise ValueError("provide rng or an explicit center")
        center = (rngr_control(rng) * width, rngr_control(rng) * height)
    left = int(math.floor(center[0] - (box_width - 1) / 2 + 0.5))
    top = int(math.floor(center[1] - (box_height - 1) / 2 + 0.5))
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + box_width), min(height, top + box_height)
    if x0 >= x1 or y0 >= y1:
        return background
    local = (slice(y0 - top, y1 - top), slice(x0 - left, x1 - left))
    mask = shape[local]
    if mask.ndim == 2:
        mask = mask[..., None]
    alpha = mask * opacity
    region = background[y0:y1, x0:x1]
    region *= 1 - alpha
    region += fill[local] * alpha
    return background
