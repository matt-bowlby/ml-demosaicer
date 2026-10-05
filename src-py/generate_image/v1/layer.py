import numpy as np
from numpy.random import Generator


def layer(background, shape, fill, opacity=1.0, *, rng: Generator | None = None,
          center: tuple[float, float] | None = None) -> np.ndarray:
    """Blend a local shape in place, using its bounding-box center as origin.

    Supply an RNG for random placement, or an explicit (x, y) center.
    Shapes extending past the image edges are clipped during compositing.
    """
    height, width = background.shape[:2]
    box_height, box_width = shape.shape[:2]
    if fill.shape != (box_height, box_width, 3):
        raise ValueError("fill must match the shape's bounding box and have three channels")
    if center is None:
        if rng is None:
            raise ValueError("provide rng or an explicit center")
        center = (float(rng.uniform(0, width)), float(rng.uniform(0, height)))
    left = int(np.floor(center[0] - (box_width - 1) / 2 + 0.5))
    top = int(np.floor(center[1] - (box_height - 1) / 2 + 0.5))
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + box_width), min(height, top + box_height)
    if x0 >= x1 or y0 >= y1:
        return background
    local = np.s_[y0 - top:y1 - top, x0 - left:x1 - left]
    mask = shape[local]
    if mask.ndim == 2:
        mask = mask[..., None]
    alpha = mask * opacity
    region = background[y0:y1, x0:x1]
    region *= 1 - alpha
    region += fill[local] * alpha
    return background
