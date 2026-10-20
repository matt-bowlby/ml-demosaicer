"""RGB fills in [0, 1], returned as (height, width, 3) float32 device tensors.

Width and height describe the local shape bounding box, not the output image.
Pixel coordinates start at (0, 0) within that box. Angles are counterclockwise radians in
image coordinates. Repeating patterns rotate about the origin; their offsets
are measured along the rotated local axes in pixels. Phase is in radians.
"""

import math
import numpy as np
import torch
from opensimplex import OpenSimplex

from device import device
from .random import RandomStreams


def solid(color: torch.Tensor, width: int, height: int) -> torch.Tensor:
    return torch.broadcast_to(color, (height, width, 3)).clone()


def linear_gradient(
    points: torch.Tensor, colors: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    """Project onto the line between two points; clamp past either endpoint."""
    grid_x, grid_y = _grid(width, height)
    direction = points[1] - points[0]
    length_squared = torch.dot(direction, direction)
    t = (
        (grid_x - points[0, 0]) * direction[0] + (grid_y - points[0, 1]) * direction[1]
    ) / length_squared
    return _blend(t, colors)


def radial_gradient(
    center: torch.Tensor,
    radii: torch.Tensor,
    angle: torch.Tensor,
    stops: torch.Tensor,
    colors: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    grid_x, grid_y = _grid(width, height)

    local_x, local_y = _rotate(grid_x - center[0], grid_y - center[1], angle)
    distance = torch.hypot(local_x / radii[0], local_y / radii[1])
    upper = torch.searchsorted(stops, distance, side="right")
    torch.clip(upper, 1, len(stops) - 1, out=upper)
    lower = upper - 1

    weights = (distance - stops[lower]) / (stops[upper] - stops[lower])
    torch.clip(weights, 0, 1, out=weights)

    start = colors[lower]
    return start + weights[..., None] * (colors[upper] - start)


def stripes(
    stripe_widths: torch.Tensor,
    angle: torch.Tensor,
    offset: torch.Tensor,
    colors: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Repeat one stripe per color; positive offset shifts the pattern forward."""
    grid_x, grid_y = _grid(width, height)
    local_x, _ = _rotate(grid_x, grid_y, angle)
    edges = torch.cumsum(stripe_widths, 0)
    position = (local_x - offset) % edges[-1]
    indices = torch.searchsorted(edges, position, side="right")
    indices %= len(colors)
    return colors[indices]


def checkerboard(
    cell_size: torch.Tensor,
    angle: torch.Tensor,
    offset: torch.Tensor,
    colors: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Alternate two colors on a rotated grid of rectangular cells."""
    grid_x, grid_y = _grid(width, height)
    local_x, local_y = _rotate(grid_x, grid_y, angle)
    parity = (
        torch.floor((local_x - offset[0]) / cell_size[0])
        + torch.floor((local_y - offset[1]) / cell_size[1])
    ) % 2
    return colors[parity.to(dtype=torch.long)]


def sinusoidal_grating(
    wavelength: torch.Tensor,
    angle: torch.Tensor,
    phase: torch.Tensor,
    colors: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Smooth periodic interpolation; zero phase starts at the color midpoint."""
    grid_x, grid_y = _grid(width, height)
    local_x, _ = _rotate(grid_x, grid_y, angle)
    t = 0.5 + 0.5 * torch.sin(2 * torch.pi * local_x / wavelength + phase)
    return _blend(t, colors)


def noise(
    stddev: torch.Tensor,
    mean_color: torch.Tensor,
    rng: RandomStreams,
    width: int,
    height: int,
) -> torch.Tensor:
    """Independent Gaussian noise per RGB channel, clipped to [0, 1]."""
    values = torch.randn(
        (height, width, 3), dtype=torch.float32, device=device, generator=rng.device_rng
    )
    values *= stddev
    values += mean_color
    torch.clip(values, 0, 1, out=values)
    return values


def smooth_noise(
    scale: torch.Tensor,
    octaves: int,
    persistence: float | torch.Tensor,
    colors: torch.Tensor,
    seed: int,
    width: int,
    height: int,
) -> torch.Tensor:
    """Layer seeded OpenSimplex noise and map it between two RGB colors.

    scale is the base feature size in pixels. Each octave doubles frequency
    and multiplies amplitude by persistence (in [0, 1]). Normalize by summed
    amplitudes, not image extrema, to preserve contrast across image sizes.
    """
    generator = OpenSimplex(seed)
    cpu_scale = scale.detach().cpu().item()
    cpu_persistence = (
        persistence.detach().cpu().item()
        if isinstance(persistence, torch.Tensor)
        else persistence
    )
    coords_x = np.arange(width, dtype=np.float32) / cpu_scale
    coords_y = np.arange(height, dtype=np.float32) / cpu_scale
    values = np.zeros((height, width), dtype=np.float32)
    amplitude, total_amplitude, frequency = 1.0, 0.0, 1.0
    for _ in range(octaves):
        # OpenSimplex returns float64; convert each octave at the library boundary.
        octave = generator.noise2array(
            coords_x * frequency, coords_y * frequency
        ).astype(np.float32)
        octave *= amplitude
        values += octave
        total_amplitude += amplitude
        amplitude *= cpu_persistence
        frequency *= 2
        if amplitude == 0:
            break
    values *= 0.5
    values /= total_amplitude
    values += 0.5
    return _blend(torch.from_numpy(values).to(device=colors.device), colors)


def _grid(width, height) -> tuple[torch.Tensor, torch.Tensor]:
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    return grid_x, grid_y


def _rotate(grid_x, grid_y, angle):
    c, s = torch.cos(angle), torch.sin(angle)
    return c * grid_x - s * grid_y, s * grid_x + c * grid_y


def _blend(t, colors):
    # Callers already provide float32 tensors on the configured device.
    weights = t.clamp(0, 1)[..., None]
    return colors[0] + weights * (colors[1] - colors[0])
