import numpy as np
from numpy.random import Generator

from .v1.generate_image import generate_image as generate_image_v1

def generate_image(rng: Generator, width: int, height: int, version: int) -> np.ndarray:
    img = None
    match version:
        case 1:
            img = generate_image_v1(rng, width, height)

    if img is None:
        raise ValueError(f"Invalid image generation version: {version}")

    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    return img


def bayer_mosaic(src_image: np.ndarray, pattern_type: str) -> np.ndarray:
    """Create a sparse (3, height, width) RGB Bayer image; unsampled colors are zero."""
    bayer_patterns = {
        "RGGB": ((0, 1), (1, 2)),
        "BGGR": ((2, 1), (1, 0)),
        "GRBG": ((1, 0), (2, 1)),
        "GBRG": ((1, 2), (0, 1)),
    }

    pattern = bayer_patterns.get(pattern_type)
    if pattern is None:
        raise ValueError(
            "Invalid Bayer pattern. Valid patterns are RGGB, BGGR, GRBG, and GBRG."
        )
    if src_image.ndim != 3 or src_image.shape[0] != 3:
        raise ValueError("RGB image must have shape (3, height, width).")

    mosaic = np.zeros_like(src_image)
    for row in range(2):
        for col in range(2):
            channel = pattern[row][col]
            mosaic[channel, row::2, col::2] = src_image[channel, row::2, col::2]
    return mosaic


def to_HWC(img: np.ndarray) -> np.ndarray:
    return img.transpose(1, 2, 0)

def from_bayer_mosaic(mosaic: np.ndarray, pattern_type: str) -> np.ndarray:
    """Convert a legacy packed (4, H/2, W/2) Bayer mosaic to sparse CHW RGB."""
    bayer_patterns = {
        "RGGB": ((0, 1), (1, 2)),
        "BGGR": ((2, 1), (1, 0)),
        "GRBG": ((1, 0), (2, 1)),
        "GBRG": ((1, 2), (0, 1)),
    }
    output_indices = {
        "RGGB": [0, 1, 2, 3],
        "BGGR": [3, 2, 1, 0],
        "GRBG": [1, 0, 3, 2],
        "GBRG": [2, 3, 0, 1],
    }

    pattern = bayer_patterns.get(pattern_type)
    indices = output_indices.get(pattern_type)
    if pattern is None or indices is None:
        raise ValueError(
            "Invalid Bayer pattern. Valid patterns are RGGB, BGGR, GRBG, and GBRG."
        )
    if mosaic.ndim != 3 or mosaic.shape[0] != 4:
        raise ValueError("Bayer mosaic must have shape (4, height, width).")

    _, height, width = mosaic.shape
    image = np.zeros((3, height * 2, width * 2), dtype=mosaic.dtype)
    image[pattern[0][0], 0::2, 0::2] = mosaic[indices[0]]
    image[pattern[0][1], 0::2, 1::2] = mosaic[indices[1]]
    image[pattern[1][0], 1::2, 0::2] = mosaic[indices[2]]
    image[pattern[1][1], 1::2, 1::2] = mosaic[indices[3]]
    return image
