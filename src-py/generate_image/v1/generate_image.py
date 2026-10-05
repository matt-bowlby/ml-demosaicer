import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt
from enum import Enum, auto

from .random import rngd, rngr, rngd_shape, rngr_shape
from .shapes import *
from .fills import *
from .layer import layer

class ColorType(Enum):
    RANDOM = auto()
    EVEN = auto()
    DARK = auto()
    NEUTRAL = auto()
    BRIGHT = auto()
    HIGH_CONTRAST = auto()

def generate_image(rng: Generator, width: int, height: int,
                   min_size: float = 1.0, max_size: float = 200.0) -> np.ndarray:
    """Generate local shapes using explicit pixel-radius limits, then place them."""
    if not (np.isfinite(min_size) and np.isfinite(max_size) and 0 < min_size <= max_size):
        raise ValueError("size limits must satisfy 0 < min_size <= max_size")
    img = np.zeros((height, width, 3), dtype=np.float32)
    image_type = list(ColorType)[int(rng.integers(len(ColorType)))]
    # Background fills cover the image directly; they need no rectangle mask.
    for _ in range(int(rng.integers(1, 8))):
        fill = _random_fill(image_type, rng, width, height, min_size, max_size)
        opacity = rngd(rng, 0.13, 14.1)
        img *= 1 - opacity
        img += fill * opacity

    num_shapes = int(rngd(rng, 0.5, 1.5) * max(width, height) / 8)
    for _ in range(num_shapes):
        shape = _random_shape(rng, min_size, max_size)
        box_height, box_width = shape.shape[:2]
        fill = _random_fill(image_type, rng, box_width, box_height, min_size, max_size)
        layer(img, shape, fill, rngd(rng, 0.3, 2.0), rng=rng)
    return img


def _random_even_color(rng: Generator) -> np.ndarray:
    return rng.random((3,), dtype=np.float32)


def _random_dark_color(rng: Generator) -> np.ndarray:
    return np.array([rngd(rng, 0.25, 8), rngd(rng, 0.25, 8), rngd(rng, 0.25, 8)], dtype=np.float32)


def _random_neutral_color(rng: Generator) -> np.ndarray:
    return np.array([rngd(rng, 0.5, 8), rngd(rng, 0.5, 8), rngd(rng, 0.5, 8)], dtype=np.float32)


def _random_high_contrast_color(rng: Generator) -> np.ndarray:
    pick = rng.integers(0, 5)
    if pick < 2:
        return _random_dark_color(rng)
    if pick < 5:
        return _random_bright_color(rng)
    return _random_neutral_color(rng)


def _random_color(rng: Generator) -> np.ndarray:
    match rng.integers(0, 4):
        case 0:
            return _random_even_color(rng)
        case 1:
            return _random_dark_color(rng)
        case 2:
            return _random_neutral_color(rng)
        case 3:
            return _random_bright_color(rng)

    return _random_high_contrast_color(rng)


def _random_color_from(type: ColorType, rng: Generator, num: int = 1) -> np.ndarray:
    match (type):
        case ColorType.RANDOM:
            return np.array([
                _random_color(rng) for i in range(num)
            ], dtype=np.float32)
        case ColorType.EVEN:
            return np.array([
                _random_even_color(rng) for i in range(num)
            ], dtype=np.float32)
        case ColorType.DARK:
            return np.array([
                _random_dark_color(rng) for i in range(num)
            ], dtype=np.float32)
        case ColorType.NEUTRAL:
            return np.array([
                _random_neutral_color(rng) for i in range(num)
            ], dtype=np.float32)
        case ColorType.BRIGHT:
            return np.array([
                _random_bright_color(rng) for i in range(num)
            ], dtype=np.float32)
        case ColorType.HIGH_CONTRAST:
            return np.array([
                _random_high_contrast_color(rng) for i in range(num)
            ], dtype=np.float32)


def _random_pattern_size(rng: Generator, maximum: float, minimum: float = 1.0) -> float:
    return minimum + rngd(rng, 0.01, 9.7) * max(0.0, maximum - minimum)

def _random_bright_color(rng: Generator) -> np.ndarray:
    return np.array([rngd(rng, 0.75, 8), rngd(rng, 0.75, 8), rngd(rng, 0.75, 8)], dtype=np.float32)


def _random_radius(rng: Generator, min_size: float, max_size: float) -> float:
    """Shared maximum distance from shape center to its outer boundary."""
    return min_size + rngd(rng, 0.05, 8.8) * (max_size - min_size)


def _random_axes(rng: Generator, min_size: float, max_size: float) -> tuple[float, float]:
    axes = np.array([rngd(rng, 0.05, 8.8), rngd(rng, 0.5, 8.8)], dtype=np.float32)
    axes *= _random_radius(rng, min_size, max_size) / axes.max()
    return float(axes[0]), float(axes[1])


def _random_vertex_shape(rng: Generator, points: np.ndarray, min_size: float, max_size: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    points = points - points.mean(axis=0)
    points *= _random_radius(rng, min_size, max_size) / np.linalg.norm(points, axis=1).max()
    return polygon(
        points, angle=rngr(rng) * 2 * np.pi,
    )


def _random_circle(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    rx, ry = _random_axes(rng, min_size, max_size)
    return circle(
        rx=rx,
        ry=ry,
        rotation=rngr(rng) * 2 * np.pi,
    )


def _random_ring(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    rx, ry = _random_axes(rng, min_size, max_size)
    return ring(
        rx,
        ry,
        thickness=rngr(rng) * min(rx, ry),
        rotation=rngr(rng) * 2 * np.pi,
    )


def _random_pie_slice(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    rx, ry = _random_axes(rng, min_size, max_size)
    thickness = rngr(rng) * min(rx, ry)
    if rng.random() <= 0.5:
        thickness = max(rx, ry)
    return pie_slice(
        rx=rx,
        ry=ry,
        thickness=thickness,
        start_angle=rngr(rng) * 2 * np.pi,
        end_angle=rngr(rng) * 2 * np.pi,
        rotation=rngr(rng) * 2 * np.pi,
    )


def _random_equilateral_triangle(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    return equilateral_triangle(
        radius=_random_radius(rng, min_size, max_size),
        angle=rngr(rng) * 2 * np.pi,
    )


def _random_isosceles_triangle(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    # Positive base and altitude guarantee a valid triangle.
    base, altitude = rngd(rng), rngd(rng)
    points = np.array([[-base / 2, 0], [base / 2, 0], [0, -altitude]], dtype=np.float32)
    return _random_vertex_shape(rng, points, min_size, max_size)


def _random_scalene_triangle(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    # Construct geometry directly instead of independently sampling side lengths.
    base, altitude = rngd(rng), rngd(rng)
    apex_x = rngr(rng) * base
    points = np.array([[0, 0], [base, 0], [apex_x, -altitude]], dtype=np.float32)
    return _random_vertex_shape(rng, points, min_size, max_size)


def _random_square(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    return square(
        radius=_random_radius(rng, min_size, max_size),
        angle=rngr(rng) * 2 * np.pi,
    )


def _random_rectangle(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    a, b = rngd(rng), rngd(rng)
    points = np.array([[0, 0], [a, 0], [a, -b], [0, -b]], dtype=np.float32)
    return _random_vertex_shape(rng, points, min_size, max_size)


def _random_parallelogram(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    a, b = rngd(rng), rngd(rng)
    altitude = rngd(rng) * b
    offset = np.sqrt(b**2 - altitude**2)
    points = np.array([[0, 0], [a, 0], [offset + a, -altitude], [offset, -altitude]], dtype=np.float32)
    return _random_vertex_shape(rng, points, min_size, max_size)


def _random_trapezoid(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    a, b, altitude = rngd(rng), rngd(rng), rngd(rng)
    offset = rngr(rng) * abs(a - b)
    points = np.array([[0, 0], [a, 0], [offset + b, -altitude], [offset, -altitude]], dtype=np.float32)
    return _random_vertex_shape(rng, points, min_size, max_size)


def _random_star(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    outer_radius = _random_radius(rng, min_size, max_size)
    inner_radius = rngr(rng) * outer_radius
    return star(
        inner_radius,
        outer_radius,
        points=int(rng.integers(3, 16)),
        angle=rngr(rng) * 2 * np.pi,
    )


def _random_regular_polygon(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    return regular_polygon(
        radius=_random_radius(rng, min_size, max_size),
        num_sides=int(rng.integers(3, 16)),
        angle=rngr(rng) * 2 * np.pi,
    )


def _random_polygon(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    """Draw 3-15 vertices without crossings; convex and concave outlines are possible."""
    points = (rng.random((rng.integers(3, 16), 2), dtype=np.float32) * 2 - 1)
    points -= points.mean(axis=0)
    # The mean lies inside the convex hull; angular order prevents edge crossings.
    order = np.argsort(np.arctan2(points[:, 1], points[:, 0]))
    return _random_vertex_shape(rng, points[order], min_size, max_size)


def _random_poly_line(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    """Connect 2-15 local points in their sampled order."""
    points = (rng.random((rng.integers(2, 16), 2), dtype=np.float32) * 2 - 1)
    points -= points.mean(axis=0)
    points *= _random_radius(rng, min_size, max_size) / np.linalg.norm(points, axis=1).max()
    return poly_line(points, line_width=_random_pattern_size(rng, max_size, min_size),
                     angle=rngr(rng) * 2 * np.pi)


def _random_shape(rng: Generator, min_size: float, max_size: float) -> np.ndarray:
    """Choose uniformly among the available shape generators."""
    generators = (
        _random_circle, _random_ring, _random_pie_slice,
        _random_equilateral_triangle, _random_isosceles_triangle,
        _random_scalene_triangle, _random_square, _random_rectangle,
        _random_parallelogram, _random_trapezoid, _random_star,
        _random_regular_polygon, _random_polygon, _random_poly_line,
    )
    return generators[int(rng.integers(len(generators)))](rng, min_size, max_size)

def _random_solid(type: ColorType, rng: Generator, width: int, height: int, min_size: float = 1.0, max_size: float = 200.0) -> np.ndarray:
    return solid(
        _random_color_from(type, rng)[0],
        width,
        height
    )

def _random_linear_gradient(type: ColorType, rng: Generator, width: int, height: int, min_size: float = 1.0, max_size: float = 200.0) -> np.ndarray:
    return linear_gradient(
        points=rngr_shape(rng, (2, 2)) * np.array([width, height], dtype=np.float32),
        colors=_random_color_from(type, rng, 2),
        width=width,
        height=height
    )

def _random_radial_gradient(type: ColorType, rng: Generator, width: int, height: int, min_size: float = 1.0, max_size: float = 200.0) -> np.ndarray:
    """Interpolate 2–7 colors, with stops at the center and ellipse boundary."""
    count = int(rng.integers(2, 8))
    # Resample the interior in the extremely unlikely event of duplicate stops.
    stops = np.concatenate((np.array([0], dtype=np.float32), np.sort(rng.random(count - 2, dtype=np.float32)), np.array([1], dtype=np.float32)))
    while np.any(np.diff(stops) <= 0):
        stops[1:-1] = np.sort(rng.random(count - 2, dtype=np.float32))
    return radial_gradient(
        center=rngr_shape(rng, (2,)) * np.array([width, height], dtype=np.float32),
        radii=min_size + rngd_shape(rng, (2,)) * (max_size - min_size),
        angle=rngr(rng) * 2 * np.pi,
        stops=stops,
        colors=_random_color_from(type, rng, count),
        width=width,
        height=height,
    )


def _random_stripes(type: ColorType, rng: Generator, width: int, height: int, min_size: float = 1.0, max_size: float = 200.0) -> np.ndarray:
    """Repeat 2–7 colors with independently sampled stripe widths."""
    count = int(rng.integers(2, 8))
    widths = np.array([_random_pattern_size(rng, max_size, min_size) for _ in range(count)], dtype=np.float32)
    return stripes(
        stripe_widths=widths,
        angle=rngr(rng) * 2 * np.pi,
        offset=float(rngr(rng) * widths.sum()),
        colors=_random_color_from(type, rng, count),
        width=width,
        height=height,
    )


def _random_checkerboard(type: ColorType, rng: Generator, width: int, height: int, min_size: float = 1.0, max_size: float = 200.0) -> np.ndarray:
    """Alternate two colors with rectangular cells and random rotation/offset."""
    cell_size = np.array([
        _random_pattern_size(rng, max_size, min_size),
        _random_pattern_size(rng, max_size, min_size),
    ], dtype=np.float32)
    return checkerboard(
        cell_size=cell_size,
        angle=rngr(rng) * 2 * np.pi,
        offset=rng.random(2, dtype=np.float32) * (2 * cell_size),
        colors=_random_color_from(type, rng, 2),
        width=width,
        height=height,
    )


def _random_sinusoidal_grating(type: ColorType, rng: Generator, width: int, height: int, min_size: float = 1.0, max_size: float = 200.0) -> np.ndarray:
    return sinusoidal_grating(
        wavelength=_random_pattern_size(rng, max_size, minimum=min_size),
        angle=rngr(rng) * 2 * np.pi,
        phase=rngr(rng) * 2 * np.pi,
        colors=_random_color_from(type, rng, 2),
        width=width,
        height=height,
    )


def _random_noise(type: ColorType, rng: Generator, width: int, height: int, min_size: float = 1.0, max_size: float = 200.0) -> np.ndarray:
    return noise(
        stddev=rngd(rng) * 0.5,
        mean_color=_random_color_from(type, rng)[0],
        rng=rng,
        width=width,
        height=height,
    )


def _random_smooth_noise(type: ColorType, rng: Generator, width: int, height: int, min_size: float = 1.0, max_size: float = 200.0) -> np.ndarray:
    return smooth_noise(
        scale=_random_pattern_size(rng, max_size, min_size),
        octaves=int(rng.integers(1, 7)),
        persistence=float(rng.uniform(0.3, 0.8)),
        colors=_random_color_from(type, rng, 2),
        seed=int(rng.integers(0, 2**31)),
        width=width,
        height=height,
    )

def _random_fill(type: ColorType, rng: Generator, width: int, height: int, min_size: float = 1.0, max_size: float = 200.0) -> np.ndarray:
    """Choose uniformly among fill generators using the requested color type."""
    generators = (
        _random_solid, _random_linear_gradient, _random_radial_gradient,
        _random_stripes, _random_checkerboard, _random_sinusoidal_grating,
        _random_noise, _random_smooth_noise,
    )
    return generators[int(rng.integers(len(generators)))](type, rng, width, height, min_size, max_size)
