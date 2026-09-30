"""RGB fills in [0, 1], returned as (height, width, 3) float32 arrays.

Pixel coordinates start at (0, 0). Angles are counterclockwise radians in
image coordinates. Repeating patterns rotate about the origin; their offsets
are measured along the rotated local axes in pixels. Phase is in radians.
"""

import numpy as np
from opensimplex import OpenSimplex

def solid(color: np.ndarray, width: int, height: int) -> np.ndarray:
    _dimensions(width, height)
    color = _colors(color, (3,))
    return np.broadcast_to(color, (height, width, 3)).astype(np.float32).copy()

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
    stops = np.asarray(stops, dtype=float)
    if (stops.ndim != 1 or len(stops) < 2 or not np.all(np.isfinite(stops))
            or np.any(stops < 0) or np.any(stops > 1) or np.any(np.diff(stops) <= 0)):
        raise ValueError("stops must contain at least two increasing values in [0, 1]")
    if np.any(radii <= 0):
        raise ValueError("radii must be positive")
    colors = _colors(colors, (len(stops), 3))
    grid_x, grid_y = _grid(width, height)
    local_x, local_y = _rotate(grid_x - center[0], grid_y - center[1], angle)
    distance = np.hypot(local_x / radii[0], local_y / radii[1])
    return np.stack([np.interp(distance, stops, colors[:, c]) for c in range(3)], axis=-1).astype(np.float32)

def stripes(stripe_widths: np.ndarray, angle: float, offset: float, colors: np.ndarray, width: int, height: int) -> np.ndarray:
    """Repeat one stripe per color; positive offset shifts the pattern forward."""
    stripe_widths = np.asarray(stripe_widths, dtype=float)
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
    return colors[indices].astype(np.float32)

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
    return colors[parity.astype(np.intp)].astype(np.float32)

def sinusoidal_grating(wavelength: float, angle: float, phase: float, colors: np.ndarray, width: int, height: int) -> np.ndarray:
    """Smooth periodic interpolation; zero phase starts at the color midpoint."""
    _positive(wavelength, "wavelength")
    _finite(phase, "phase")
    colors = _colors(colors, (2, 3))
    grid_x, grid_y = _grid(width, height)
    local_x, _ = _rotate(grid_x, grid_y, angle)
    t = 0.5 + 0.5 * np.sin(2 * np.pi * local_x / wavelength + phase)
    return _blend(t, colors)

def noise(stddev: float, mean_color: np.ndarray, seed: int, width: int, height: int) -> np.ndarray:
    """Independent Gaussian noise per RGB channel, clipped to [0, 1]."""
    _dimensions(width, height)
    _finite(stddev, "stddev")
    if stddev < 0:
        raise ValueError("stddev must be nonnegative")
    mean_color = _colors(mean_color, (3,))
    rng = np.random.default_rng(seed)
    values = rng.normal(mean_color, stddev, size=(height, width, 3))
    return np.clip(values, 0, 1).astype(np.float32)

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
    coords_x = np.arange(width, dtype=float) / scale
    coords_y = np.arange(height, dtype=float) / scale
    values = np.zeros((height, width), dtype=float)
    amplitude, total_amplitude, frequency = 1.0, 0.0, 1.0
    for _ in range(octaves):
        values += amplitude * generator.noise2array(coords_x * frequency, coords_y * frequency)
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
    value = np.asarray(value, dtype=float)
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
    grid_y, grid_x = np.indices((height, width), dtype=float)
    return grid_x, grid_y


def _rotate(grid_x, grid_y, angle):
    _finite(angle, "angle")
    c, s = np.cos(angle), np.sin(angle)
    return c * grid_x - s * grid_y, s * grid_x + c * grid_y


def _blend(t, colors):
    t = np.clip(t, 0, 1)[..., None]
    return ((1 - t) * colors[0] + t * colors[1]).astype(np.float32)
