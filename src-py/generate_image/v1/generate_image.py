import torch
import numpy as np
from enum import Enum, auto

from device import device

from .random import (
    RandomStreams, rngd, rngr, rngd_shape, rngr_shape, rngi,
    rngd_control, rngr_control, rngr_control_shape,
)
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


def generate_image(
    rng: RandomStreams,
    width: int,
    height: int,
) -> torch.Tensor:
    """Composite device tensors and return an HWC tensor.

    RandomStreams supplies device randomness for rendering and CPU randomness
    for counts, choices, and placement.
    """
    min_size = torch.tensor(1.0, dtype=torch.float32, device=device)
    max_size = torch.tensor(min(width, height), dtype=torch.float32, device=device)
    img = torch.zeros((height, width, 3), dtype=torch.float32, device=device)
    image_type = list(ColorType)[rngi(rng, len(ColorType))]
    # Background fills cover the image directly; they need no rectangle mask.
    for _ in range(rngi(rng, 1, 8)):
        fill = _random_fill(image_type, rng, width, height, min_size, max_size)
        opacity = rngd(rng, 0.13, 14.1)
        img *= 1 - opacity
        img += fill * opacity

    num_shapes = int(rngd_control(rng, 0.5, 1.5) * (max(max(width, height) / 8, 10)))
    for _ in range(num_shapes):
        shape = _random_shape(rng, min_size, max_size)
        box_height, box_width = shape.shape[:2]
        fill = _random_fill(image_type, rng, box_width, box_height, min_size, max_size)
        layer(img, shape, fill, rngd(rng, 0.3, 2.0), rng=rng)
    return img


def _random_even_color(rng: RandomStreams) -> torch.Tensor:
    return rngr_shape(rng, (3,))


def _random_dark_color(rng: RandomStreams) -> torch.Tensor:
    return torch.stack([rngd(rng, 0.25, 8), rngd(rng, 0.25, 8), rngd(rng, 0.25, 8)])


def _random_neutral_color(rng: RandomStreams) -> torch.Tensor:
    return torch.stack([rngd(rng, 0.5, 8), rngd(rng, 0.5, 8), rngd(rng, 0.5, 8)])


def _random_high_contrast_color(rng: RandomStreams) -> torch.Tensor:
    pick = rngi(rng, 0, 5)
    if pick < 2:
        return _random_dark_color(rng)
    if pick < 5:
        return _random_bright_color(rng)
    return _random_neutral_color(rng)


def _random_color(rng: RandomStreams) -> torch.Tensor:
    match rngi(rng, 0, 4):
        case 0:
            return _random_even_color(rng)
        case 1:
            return _random_dark_color(rng)
        case 2:
            return _random_neutral_color(rng)
        case 3:
            return _random_bright_color(rng)

    return _random_high_contrast_color(rng)


def _random_color_from(type: ColorType, rng: RandomStreams, num: int) -> torch.Tensor:
    match (type):
        case ColorType.RANDOM:
            return torch.stack([_random_color(rng) for _ in range(num)])
        case ColorType.EVEN:
            return torch.stack([_random_even_color(rng) for _ in range(num)])
        case ColorType.DARK:
            return torch.stack([_random_dark_color(rng) for _ in range(num)])
        case ColorType.NEUTRAL:
            return torch.stack([_random_neutral_color(rng) for _ in range(num)])
        case ColorType.BRIGHT:
            return torch.stack([_random_bright_color(rng) for _ in range(num)])
        case ColorType.HIGH_CONTRAST:
            return torch.stack([_random_high_contrast_color(rng) for _ in range(num)])


def _random_pattern_size(
    rng: RandomStreams, maximum: torch.Tensor, minimum: torch.Tensor
) -> torch.Tensor:
    return minimum + rngd(rng, 0.01, 9.7) * (maximum - minimum).clamp_min(0)


def _random_bright_color(rng: RandomStreams) -> torch.Tensor:
    return torch.stack((rngd(rng, 0.75, 8), rngd(rng, 0.75, 8), rngd(rng, 0.75, 8)))


def _random_radius(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    """Shared maximum distance from shape center to its outer boundary."""
    return min_size + rngd(rng, 0.05, 8.8) * (max_size - min_size)


def _random_axes(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    axes = torch.stack([rngd(rng, 0.05, 8.8), rngd(rng, 0.5, 8.8)])
    axes *= _random_radius(rng, min_size, max_size) / axes.max()
    return axes[0], axes[1]


def _random_vertex_shape(
    rng: RandomStreams,
    points: torch.Tensor,
    min_size: torch.Tensor,
    max_size: torch.Tensor,
) -> torch.Tensor:
    points = points - points.mean(0)
    points *= (
        _random_radius(rng, min_size, max_size)
        / torch.linalg.norm(points, axis=1).max()
    )
    return polygon(
        points,
        angle=rngr(rng) * 2 * torch.pi,
    )


def _random_circle(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    rx, ry = _random_axes(rng, min_size, max_size)
    return circle(
        rx=rx,
        ry=ry,
        rotation=rngr(rng) * 2 * np.pi,
    )


def _random_ring(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    rx, ry = _random_axes(rng, min_size, max_size)
    return ring(
        rx,
        ry,
        thickness=rngr(rng) * torch.minimum(rx, ry),
        rotation=rngr(rng) * 2 * np.pi,
    )


def _random_pie_slice(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    rx, ry = _random_axes(rng, min_size, max_size)
    thickness = rngr(rng) * torch.minimum(rx, ry)
    if rngr_control(rng) <= 0.5:
        thickness = torch.maximum(rx, ry)
    return pie_slice(
        rx=rx,
        ry=ry,
        thickness=thickness,
        start_angle=rngr(rng) * 2 * np.pi,
        end_angle=rngr(rng) * 2 * np.pi,
        rotation=rngr(rng) * 2 * np.pi,
    )


def _random_equilateral_triangle(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    return equilateral_triangle(
        radius=_random_radius(rng, min_size, max_size),
        angle=rngr(rng) * 2 * np.pi,
    )


def _random_isosceles_triangle(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    # Positive base and altitude guarantee a valid triangle.
    base, altitude = rngd(rng), rngd(rng)
    points = torch.stack((
        torch.stack((-base / 2, base.new_zeros(()))),
        torch.stack((base / 2, base.new_zeros(()))),
        torch.stack((base.new_zeros(()), -altitude)),
    ))
    return _random_vertex_shape(rng, points, min_size, max_size)


def _random_scalene_triangle(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    # Construct geometry directly instead of independently sampling side lengths.
    base, altitude = rngd(rng), rngd(rng)
    apex_x = rngr(rng) * base
    points = torch.stack((
        base.new_zeros(2),
        torch.stack((base, base.new_zeros(()))),
        torch.stack((apex_x, -altitude)),
    ))
    return _random_vertex_shape(rng, points, min_size, max_size)


def _random_square(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    return square(
        radius=_random_radius(rng, min_size, max_size),
        angle=rngr(rng) * 2 * np.pi,
    )


def _random_rectangle(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    a, b = rngd(rng), rngd(rng)
    points = torch.stack((
        a.new_zeros(2),
        torch.stack((a, a.new_zeros(()))),
        torch.stack((a, -b)),
        torch.stack((a.new_zeros(()), -b)),
    ))
    return _random_vertex_shape(rng, points, min_size, max_size)


def _random_parallelogram(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    a, b = rngd(rng), rngd(rng)
    altitude = rngd(rng) * b
    offset = torch.sqrt(b**2 - altitude**2)
    points = torch.stack((
        a.new_zeros(2),
        torch.stack((a, a.new_zeros(()))),
        torch.stack((offset + a, -altitude)),
        torch.stack((offset, -altitude)),
    ))
    return _random_vertex_shape(rng, points, min_size, max_size)


def _random_trapezoid(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    a, b, altitude = rngd(rng), rngd(rng), rngd(rng)
    offset = rngr(rng) * abs(a - b)
    points = torch.stack((
        a.new_zeros(2),
        torch.stack((a, a.new_zeros(()))),
        torch.stack((offset + b, -altitude)),
        torch.stack((offset, -altitude)),
    ))
    return _random_vertex_shape(rng, points, min_size, max_size)


def _random_star(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    outer_radius = _random_radius(rng, min_size, max_size)
    inner_radius = rngr(rng) * outer_radius
    return star(
        inner_radius,
        outer_radius,
        points=rngi(rng, 3, 16),
        angle=rngr(rng) * 2 * torch.pi,
    )


def _random_regular_polygon(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    return regular_polygon(
        radius=_random_radius(rng, min_size, max_size),
        num_sides=rngi(rng, 3, 16),
        angle=rngr(rng) * 2 * np.pi,
    )


def _random_polygon(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    """Draw 3-15 vertices without crossings; convex and concave outlines are possible."""
    points = rngr_shape(rng, (rngi(rng, 3, 16), 2)) * 2 - 1
    points -= points.mean(dim=0)
    # The mean lies inside the convex hull; angular order prevents edge crossings.
    order = torch.argsort(torch.arctan2(points[:, 1], points[:, 0]))
    return _random_vertex_shape(rng, points[order], min_size, max_size)


def _random_poly_line(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    """Connect 2-15 local points in their sampled order."""
    points = rngr_shape(rng, (rngi(rng, 2, 16), 2)) * 2 - 1
    points -= points.mean(dim=0)
    points *= (
        _random_radius(rng, min_size, max_size)
        / torch.linalg.norm(points, axis=1).max()
    )
    return poly_line(
        points,
        line_width=_random_pattern_size(rng, max_size, min_size),
        angle=rngr(rng) * 2 * np.pi,
    )


def _random_shape(
    rng: RandomStreams, min_size: torch.Tensor, max_size: torch.Tensor
) -> torch.Tensor:
    """Choose uniformly among the available shape generators."""
    generators = (
        _random_circle,
        _random_ring,
        _random_pie_slice,
        _random_equilateral_triangle,
        _random_isosceles_triangle,
        _random_scalene_triangle,
        _random_square,
        _random_rectangle,
        _random_parallelogram,
        _random_trapezoid,
        _random_star,
        _random_star,
        _random_regular_polygon,
        _random_polygon,
        _random_polygon,
        _random_poly_line,
        _random_poly_line,
    )
    return generators[rngi(rng, len(generators))](rng, min_size, max_size)


def _random_solid(
    type: ColorType,
    rng: RandomStreams,
    width: int,
    height: int,
    min_size: torch.Tensor,
    max_size: torch.Tensor,
) -> torch.Tensor:
    return solid(
        _random_color_from(type, rng, 1)[0],
        width,
        height,
    )


def _random_linear_gradient(
    type: ColorType,
    rng: RandomStreams,
    width: int,
    height: int,
    min_size: torch.Tensor,
    max_size: torch.Tensor,
) -> torch.Tensor:
    return linear_gradient(
        points=rngr_shape(rng, (2, 2))
        * torch.tensor([width, height], dtype=torch.float32, device=device),
        colors=_random_color_from(type, rng, 2),
        width=width,
        height=height,
    )


def _random_radial_gradient(
    type: ColorType,
    rng: RandomStreams,
    width: int,
    height: int,
    min_size: torch.Tensor,
    max_size: torch.Tensor,
) -> torch.Tensor:
    """Interpolate 2–7 colors, with stops at the center and ellipse boundary."""
    count = rngi(rng, 2, 8)

    interior = rngr_control_shape(rng, (count - 2,))
    stops = torch.cat(
        (
            interior.new_zeros(1),
            interior.sort().values,
            interior.new_ones(1),
        )
    )

    while torch.any(torch.diff(stops) <= 0):
        stops[1:-1] = rngr_control_shape(rng, (count - 2,)).sort().values
    return radial_gradient(
        center=rngr_shape(rng, (2,))
        * torch.tensor([width, height], dtype=torch.float32, device=device),
        radii=min_size + rngd_shape(rng, (2,)) * (max_size - min_size),
        angle=rngr(rng) * 2 * torch.pi,
        stops=stops.to(device),
        colors=_random_color_from(type, rng, count),
        width=width,
        height=height,
    )


def _random_stripes(
    type: ColorType,
    rng: RandomStreams,
    width: int,
    height: int,
    min_size: torch.Tensor,
    max_size: torch.Tensor,
) -> torch.Tensor:
    """Repeat 2–7 colors with independently sampled stripe widths."""
    count = rngi(rng, 2, 8)
    widths = torch.stack(
        [_random_pattern_size(rng, max_size, min_size) for _ in range(count)]
    )
    return stripes(
        stripe_widths=widths,
        angle=rngr(rng) * 2 * torch.pi,
        offset=rngr(rng) * widths.sum(),
        colors=_random_color_from(type, rng, count),
        width=width,
        height=height,
    )


def _random_checkerboard(
    type: ColorType,
    rng: RandomStreams,
    width: int,
    height: int,
    min_size: torch.Tensor,
    max_size: torch.Tensor,
) -> torch.Tensor:
    """Alternate two colors with rectangular cells and random rotation/offset."""
    cell_size = torch.stack(
        [
            _random_pattern_size(rng, max_size, min_size),
            _random_pattern_size(rng, max_size, min_size),
        ]
    )

    return checkerboard(
        cell_size=cell_size,
        angle=rngr(rng) * 2 * torch.pi,
        offset=rngr_shape(rng, 2) * (2 * cell_size),
        colors=_random_color_from(type, rng, 2),
        width=width,
        height=height,
    )


def _random_sinusoidal_grating(
    type: ColorType,
    rng: RandomStreams,
    width: int,
    height: int,
    min_size: torch.Tensor,
    max_size: torch.Tensor,
) -> torch.Tensor:
    return sinusoidal_grating(
        wavelength=_random_pattern_size(rng, max_size, minimum=min_size),
        angle=rngr(rng) * 2 * torch.pi,
        phase=rngr(rng) * 2 * torch.pi,
        colors=_random_color_from(type, rng, 2),
        width=width,
        height=height,
    )


def _random_noise(
    type: ColorType,
    rng: RandomStreams,
    width: int,
    height: int,
) -> torch.Tensor:
    return noise(
        stddev=rngd(rng) * 0.5,
        mean_color=_random_color_from(type, rng, 1)[0],
        rng=rng,
        width=width,
        height=height,
    )


def _random_smooth_noise(
    type: ColorType,
    rng: RandomStreams,
    width: int,
    height: int,
    min_size: torch.Tensor,
    max_size: torch.Tensor,
) -> torch.Tensor:
    return smooth_noise(
        scale=_random_pattern_size(rng, max_size, min_size),
        octaves=rngi(rng, 1, 7),
        persistence=0.3 + 0.5 * rngr_control(rng),
        colors=_random_color_from(type, rng, 2),
        seed=rngi(rng, 0, 2**31),
        width=width,
        height=height,
    )


def _random_fill(
    type: ColorType,
    rng: RandomStreams,
    width: int,
    height: int,
    min_size: torch.Tensor,
    max_size: torch.Tensor,
) -> torch.Tensor:
    """Choose uniformly among fill generators using the requested color type."""
    generators = (
        _random_solid,
        _random_linear_gradient,
        _random_radial_gradient,
        _random_stripes,
        _random_stripes,
        _random_checkerboard,
        _random_checkerboard,
        _random_sinusoidal_grating,
        _random_smooth_noise,
    )
    return generators[rngi(rng, len(generators))](
        type, rng, width, height, min_size, max_size
    )
