import numpy as np
from numpy.random import Generator

def rngd(rng: Generator, peak: float = 0.15, concentration: float = 2.2, shape = None) -> float:
    a = 1 + peak * concentration
    b = 1 + (1 - peak) * concentration
    return float(rng.beta(a, b, shape))

def rngr(rng: Generator) -> float:
    return rng.random()

def rngr_shape(rng: Generator, shape) -> np.ndarray:
    return rng.random(shape)

def rngd_shape(rng: Generator, shape, peak: float = 0.15, concentration: float = 2.2) -> np.ndarray:
    a = 1 + peak * concentration
    b = 1 + (1 - peak) * concentration
    return rng.beta(a, b, shape)