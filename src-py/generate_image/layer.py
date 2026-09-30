import numpy as np

def layer(background, shape, fill, opacity = 1.0) -> np.ndarray:
    """Blend the fill inside the shape, preserving the background outside it."""
    alpha = shape * opacity
    return alpha * fill + background * (1 - alpha)
