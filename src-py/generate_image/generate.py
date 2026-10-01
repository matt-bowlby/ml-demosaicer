import numpy as np
from numpy.random import Generator

from .v1.generate_image import generate_image as generate_image_v1

def generate_image(rng: Generator, width: int, height: int, version: int) -> np.ndarray:
    match version:
        case 1:
            return generate_image_v1(rng, width, height)
    return np.array([])
