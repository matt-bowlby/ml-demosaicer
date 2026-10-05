"""RGB fills in [0, 1], returned as (height, width, 3) float32 arrays.

Width and height describe the local shape bounding box, not the output image.
Pixel coordinates start at (0, 0) within that box. Angles are counterclockwise radians in
image coordinates. Repeating patterns rotate about the origin; their offsets
are measured along the rotated local axes in pixels. Phase is in radians.
"""

import numpy as np
from opensimplex import OpenSimplex
from numpy.random import Generator

def solid(color: np.ndarray, width: int, height: int) -> np.ndarray:
    _dimensions(width, height)
    color = _colors(color, (3,))
    return np.broadcast_to(color, (height, width, 3)).copy()

def linear_gradient(points: np.ndarray, colors: np.ndarray, width: int, height: int) -> np.ndarray:
    """Project onto the line between two points; clamp past either endpoint."""
    points = _array(points, (2, 2), "points")
    colors = _colors(colors, (2, 3))
    grid_x, grid_y = _grid(width, height)
    direction = points[1] - points[0]
    length_squared = np.dot(direction, direction)
    if length_squared == 0:
        raise ValueError("Gradient points must be distinct")
    t = ((grid_x - points[0, 0]) * direction[0]
         + (grid_y - points[0, 1]) * direction[1]) / length_squared
    return _blend(t, colors)

def radial_gradient(center: np.ndarray, radii: np.ndarray, angle: float, stops: np.ndarray, colors: np.ndarray, width: int, height: int) -> np.ndarray:
    """Interpolate color stops at fractions of a rotated ellipse's radii."""
    center = _array(center, (2,), "center")
    radii = _array(radii, (2,), "radii")
    stops = np.asarray(stops, dtype=np.float32)
    if (stops.ndim != 1 or len(stops) < 2 or not np.all(np.isfinite(stops))
            or np.any(stops < 0) or np.any(stops > 1) or np.any(np.diff(stops) <= 0)):
        raise ValueError("stops must contain at least two increasing values in [0, 1]")
    if np.any(radii <= 0):
        raise ValueError("radii must be positive")
    colors = _colors(colors, (len(stops), 3))
    grid_x, grid_y = _grid(width, height)
    local_x, local_y = _rotate(grid_x - center[0], grid_y - center[1], angle)
    distance = np.hypot(local_x / radii[0], local_y / radii[1])
    # np.interp always returns float64; interpolate the bracketing stops directly.
    upper = np.searchsorted(stops, distance, side="right")
    np.clip(upper, 1, len(stops) - 1, out=upper)
    lower = upper - 1
    weights = (distance - stops[lower]) / (stops[upper] - stops[lower])
    np.clip(weights, 0, 1, out=weights)
    result = np.empty((height, width, 3), dtype=np.float32)
    for channel in range(3):
        start = colors[lower, channel]
        result[..., channel] = start + weights * (colors[upper, channel] - start)
    return result

def stripes(stripe_widths: np.ndarray, angle: float, offset: float, colors: np.ndarray, width: int, height: int) -> np.ndarray:
    """Repeat one stripe per color; positive offset shifts the pattern forward."""
    stripe_widths = np.asarray(stripe_widths, dtype=np.float32)
    if (stripe_widths.ndim != 1 or len(stripe_widths) == 0
            or not np.all(np.isfinite(stripe_widths)) or np.any(stripe_widths <= 0)):
        raise ValueError("stripe_widths must be a nonempty vector of positive finite widths")
    colors = _colors(colors, (len(stripe_widths), 3))
    _finite(offset, "offset")
    grid_x, grid_y = _grid(width, height)
    local_x, _ = _rotate(grid_x, grid_y, angle)
    edges = np.cumsum(stripe_widths)
    position = (local_x - offset) % edges[-1]
    indices = np.searchsorted(edges, position, side="right")
    indices %= len(colors)
    return colors[indices]

def checkerboard(cell_size: np.ndarray, angle: float, offset: np.ndarray, colors: np.ndarray, width: int, height: int) -> np.ndarray:
    """Alternate two colors on a rotated grid of rectangular cells."""
    cell_size = _array(cell_size, (2,), "cell_size")
    offset = _array(offset, (2,), "offset")
    if np.any(cell_size <= 0):
        raise ValueError("cell_size must be positive")
    colors = _colors(colors, (2, 3))
    grid_x, grid_y = _grid(width, height)
    local_x, local_y = _rotate(grid_x, grid_y, angle)
    parity = (np.floor((local_x - offset[0]) / cell_size[0])
              + np.floor((local_y - offset[1]) / cell_size[1])) % 2
    return colors[parity.astype(np.intp)]

def sinusoidal_grating(wavelength: float, angle: float, phase: float, colors: np.ndarray, width: int, height: int) -> np.ndarray:
    """Smooth periodic interpolation; zero phase starts at the color midpoint."""
    _positive(wavelength, "wavelength")
    _finite(phase, "phase")
    colors = _colors(colors, (2, 3))
    grid_x, grid_y = _grid(width, height)
    local_x, _ = _rotate(grid_x, grid_y, angle)
    t = 0.5 + 0.5 * np.sin(2 * np.pi * local_x / wavelength + phase)
    return _blend(t, colors)

def noise(stddev: float, mean_color: np.ndarray, rng: Generator, width: int, height: int) -> np.ndarray:
    """Independent Gaussian noise per RGB channel, clipped to [0, 1]."""
    _dimensions(width, height)
    _finite(stddev, "stddev")
    if stddev < 0:
        raise ValueError("stddev must be nonnegative")
    mean_color = _colors(mean_color, (3,))
    # normal() has no dtype argument; standard_normal() can generate float32 directly.
    values = rng.standard_normal(size=(height, width, 3), dtype=np.float32)
    values *= np.float32(stddev)
    values += mean_color
    np.clip(values, 0, 1, out=values)
    return values

def smooth_noise(scale: float, octaves: int, persistence: float, colors: np.ndarray, seed: int, width: int, height: int) -> np.ndarray:
    """Layer seeded OpenSimplex noise and map it between two RGB colors.

    scale is the base feature size in pixels. Each octave doubles frequency
    and multiplies amplitude by persistence (in [0, 1]). Normalize by summed
    amplitudes, not image extrema, to preserve contrast across image sizes.
    """
    _dimensions(width, height)
    _positive(scale, "scale")
    if not isinstance(octaves, (int, np.integer)) or octaves < 1:
        raise ValueError("octaves must be a positive integer")
    _finite(persistence, "persistence")
    if not 0 <= persistence <= 1:
        raise ValueError("persistence must be in [0, 1]")
    colors = _colors(colors, (2, 3))
    generator = OpenSimplex(seed)
    coords_x = np.arange(width, dtype=np.float32) / scale
    coords_y = np.arange(height, dtype=np.float32) / scale
    values = np.zeros((height, width), dtype=np.float32)
    amplitude, total_amplitude, frequency = 1.0, 0.0, 1.0
    for _ in range(octaves):
        # OpenSimplex returns float64; convert each octave at the library boundary.
        octave = generator.noise2array(coords_x * frequency, coords_y * frequency).astype(np.float32)
        octave *= np.float32(amplitude)
        values += octave
        total_amplitude += amplitude
        amplitude *= persistence
        frequency *= 2
        if amplitude == 0:
            break
    return _blend(0.5 + 0.5 * values / total_amplitude, colors)


def _dimensions(width, height):
    if any(not isinstance(v, (int, np.integer)) or v <= 0 for v in (width, height)):
        raise ValueError("width and height must be positive integers")


def _array(value, shape, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return value


def _colors(value, shape):
    value = _array(value, shape, "colors")
    if np.any(value < 0) or np.any(value > 1):
        raise ValueError("RGB values must be in [0, 1]")
    return value


def _finite(value, name):
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _positive(value, name):
    _finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _grid(width, height):
    _dimensions(width, height)
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    return grid_x, grid_y


def _rotate(grid_x, grid_y, angle):
    _finite(angle, "angle")
    c, s = np.float32(np.cos(angle)), np.float32(np.sin(angle))
    return c * grid_x - s * grid_y, s * grid_x + c * grid_y


def _blend(t, colors):
    weights = np.clip(np.asarray(t, dtype=np.float32), 0, 1)
    colors = np.asarray(colors, dtype=np.float32)
    result = np.empty((*weights.shape, 3), dtype=np.float32)
    for channel in range(3):
        result[..., channel] = (colors[0, channel]
                                + weights * (colors[1, channel] - colors[0, channel]))
    return result
