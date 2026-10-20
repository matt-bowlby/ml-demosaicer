import math
from numbers import Integral
from typing import TypeAlias
from numpy.typing import NDArray
import torch
import numpy as np
from PIL import Image, ImageDraw

from device import device

Scalar: TypeAlias = float | torch.Tensor
Points: TypeAlias = NDArray[np.float32]

NORMALIZE_RGB_VALUE = 0.00392156862745098  # 1 / 255


def circle(rx: Scalar, ry: Scalar, rotation: Scalar) -> torch.Tensor:
    """Draw a rotated ellipse in its tight raster bounding box."""
    rx, ry, rotation = _cpu_scalars(rx, ry, rotation)
    local_x, local_y = _ellipse_coordinates(rx, ry, rotation)
    return _mask_image((local_x / rx) ** 2 + (local_y / ry) ** 2 <= 1)


def ring(rx: Scalar, ry: Scalar, thickness: Scalar, rotation: Scalar) -> torch.Tensor:
    """Draw an elliptical ring; thickness is subtracted from both inner radii."""
    rx, ry, thickness, rotation = _cpu_scalars(rx, ry, thickness, rotation)
    local_x, local_y = _ellipse_coordinates(rx, ry, rotation)
    return _mask_image(_ring_mask(local_x, local_y, rx, ry, thickness))


def pie_slice(
    rx: Scalar,
    ry: Scalar,
    thickness: Scalar,
    start_angle: Scalar,
    end_angle: Scalar,
    rotation: Scalar,
) -> torch.Tensor:
    """Draw an elliptical ring sector, counterclockwise from start to end.

    Angles are radians from the local positive x-axis. A span of at least
    2*pi draws a full ring; thickness >= min(rx, ry) fills to the center.
    """
    rx, ry, thickness, start_angle, end_angle, rotation = _cpu_scalars(
        rx, ry, thickness, start_angle, end_angle, rotation
    )
    local_x, local_y = _ellipse_coordinates(rx, ry, rotation)
    mask = _ring_mask(local_x, local_y, rx, ry, thickness)
    span = end_angle - start_angle
    if not math.isfinite(span):
        raise ValueError("Sector angles must be finite")
    if abs(span) < 2 * torch.pi:
        if span == 0:
            mask[:] = False
        else:
            theta = torch.atan2(-local_y / ry, local_x / rx)
            angular_mask = (theta - start_angle) % (2 * math.pi) <= span % (2 * math.pi)
            angular_mask |= (local_x == 0) & (local_y == 0)
            mask &= angular_mask
    return _mask_image(mask)


def equilateral_triangle(radius: Scalar, angle: Scalar) -> torch.Tensor:
    return regular_polygon(radius, 3, angle)


def isosceles_triangle(leg_a: Scalar, leg_bc: Scalar, angle: Scalar) -> torch.Tensor:
    leg_a, leg_bc, angle = _cpu_scalars(leg_a, leg_bc, angle)
    _positive(leg_a=leg_a, leg_bc=leg_bc)
    if leg_a >= 2 * leg_bc:
        raise ValueError("Triangle sides must satisfy the triangle inequality")
    half_base = leg_a / 2
    altitude = math.sqrt(leg_bc**2 - half_base**2)
    vertices = np.asarray(
        [
            [0, -2 * altitude / 3],
            [-half_base, altitude / 3],
            [half_base, altitude / 3],
        ],
        dtype=np.float32,
    )

    angle = float(angle)
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    rotation = np.array(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32
    )
    vertices = vertices @ rotation

    return _draw_vertices(vertices)


def scalene_triangle(
    leg_a: Scalar, leg_b: Scalar, leg_c: Scalar, angle: float
) -> torch.Tensor:
    leg_a, leg_b, leg_c, angle = _cpu_scalars(leg_a, leg_b, leg_c, angle)
    _positive(leg_a=leg_a, leg_b=leg_b, leg_c=leg_c)
    if 2 * max(leg_a, leg_b, leg_c) >= leg_a + leg_b + leg_c:
        raise ValueError("Triangle sides must satisfy the triangle inequality")
    apex_x = (leg_a**2 + leg_b**2 - leg_c**2) / (2 * leg_a)
    altitude = math.sqrt(max(0.0, leg_b**2 - apex_x**2))
    vertices = np.array(
        [
            [apex_x, -altitude],
            [0, 0],
            [leg_a, 0],
        ],
        dtype=np.float32,
    )
    vertices -= vertices.mean(0)

    angle = float(angle)
    cos_angle, sin_angle = math.cos(angle), math.sin(angle)
    rotation = np.array(
        [[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32
    )
    vertices = vertices @ rotation

    return _draw_vertices(vertices)


def square(radius: Scalar, angle: Scalar) -> torch.Tensor:
    """Draw a square with the given center-to-corner radius."""
    return regular_polygon(radius, 4, angle)


def rectangle(side_a: Scalar, side_b: Scalar, angle: Scalar) -> torch.Tensor:
    side_a, side_b, angle = _cpu_scalars(side_a, side_b, angle)
    _positive(side_a=side_a, side_b=side_b)
    points = np.array(
        [[0, 0], [side_a, 0], [side_a, -side_b], [0, -side_b]], dtype=np.float32
    )
    return polygon(points, angle)


def parallelogram(
    side_a: Scalar, side_b: Scalar, base_height: Scalar, angle: float
) -> torch.Tensor:
    """Draw a parallelogram whose upper base is shifted to the right."""
    side_a, side_b, base_height, angle = _cpu_scalars(side_a, side_b, base_height, angle)
    _positive(side_a=side_a, side_b=side_b, base_height=base_height)
    if base_height > side_b:
        raise ValueError("base_height must not exceed side_b")
    offset = math.sqrt(max(0.0, side_b**2 - base_height**2))
    return trapezoid(side_a, side_a, base_height, offset, angle)


def trapezoid(
    base_a: Scalar, base_b: Scalar, base_height: Scalar, base_offset: Scalar, angle: float
) -> torch.Tensor:
    """base_offset is the upper base's left endpoint relative to the lower one."""
    base_a, base_b, base_height, base_offset, angle = _cpu_scalars(
        base_a, base_b, base_height, base_offset, angle
    )
    _positive(base_a=base_a, base_b=base_b, base_height=base_height)
    points = np.array(
        [
            [0, 0],
            [base_a, 0],
            [base_offset + base_b, -base_height],
            [base_offset, -base_height],
        ],
        dtype=np.float32,
    )
    return polygon(points, angle)


def star(
    inner_radius: Scalar, outer_radius: Scalar, points: int, angle: Scalar
) -> torch.Tensor:
    """Draw a star with at least three points and an upward tip at zero rotation."""
    inner_radius, outer_radius, angle = _cpu_scalars(inner_radius, outer_radius, angle)
    _positive(inner_radius=inner_radius, outer_radius=outer_radius)
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius")
    if not isinstance(points, Integral) or points < 3:
        raise ValueError("points must be an integer of at least 3")
    indices = np.arange(2 * points, dtype=np.float32)
    theta = math.pi / 2 + indices * math.pi / points
    radii = np.where(indices % 2 == 0, outer_radius, inner_radius)
    vertices = np.column_stack((radii * np.cos(theta), -radii * np.sin(theta))).astype(np.float32)
    return polygon(vertices, angle)


def regular_polygon(radius: Scalar, num_sides: int, angle: Scalar) -> torch.Tensor:
    """Draw a regular polygon with a horizontal bottom edge at zero rotation."""
    radius, angle = _cpu_scalars(radius, angle)
    _positive(radius=radius)
    if not isinstance(num_sides, Integral) or num_sides < 3:
        raise ValueError("num_points must be an integer of at least 3")
    theta = (
        3 * math.pi / 2
        - math.pi / num_sides
        + np.arange(num_sides, dtype=np.float32) * 2 * math.pi / num_sides
    )
    points = np.column_stack((radius * np.cos(theta), -radius * np.sin(theta))).astype(np.float32)
    vertices = np.round(_transform_points(points, angle, 3), decimals=2)
    return _draw_vertices(vertices)


def polygon(points: torch.Tensor | Points, angle: Scalar) -> torch.Tensor:
    """Transfer geometry once for Pillow, then upload the cropped mask."""
    cpu_points, (cpu_angle,) = _cpu_geometry(points, angle)
    return _draw_vertices(_transform_points(cpu_points, cpu_angle, 3))


def poly_line(points: torch.Tensor | Points, line_width: Scalar,
              angle: Scalar) -> torch.Tensor:
    """Draw an open polyline; line width rounds to at least one pixel."""
    cpu_points, (cpu_width, cpu_angle) = _cpu_geometry(points, line_width, angle)
    if not math.isfinite(cpu_width):
        raise ValueError("line_width must be finite")
    return _draw_vertices(_transform_points(cpu_points, cpu_angle, 2), max(1, round(cpu_width)))


def _cpu_geometry(
    points: torch.Tensor | Points, *parameters: Scalar
) -> tuple[Points, list[float]]:
    """Pack device vertices and scalar parameters into a single host transfer."""
    if isinstance(points, torch.Tensor):
        points = points.detach().to(dtype=torch.float32)
        scalars = [torch.as_tensor(value, dtype=points.dtype, device=points.device).reshape(1)
                   for value in parameters]
        packed = torch.cat([points.reshape(-1), *scalars]).cpu().numpy()
        count = points.numel()
        return packed[:count].reshape(points.shape), packed[count:].tolist()
    return np.asarray(points, dtype=np.float32), _cpu_scalars(*parameters)


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _cpu_scalars(*values: Scalar) -> list[float]:
    """Move small tensor parameters to the CPU together for Pillow geometry."""
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if not tensors:
        return [float(value) for value in values]
    source_device = tensors[0].device
    return torch.stack([
        torch.as_tensor(value, dtype=torch.float32, device=source_device).detach()
        for value in values
    ]).cpu().tolist()


def _transform_points(points: Points, angle: float, minimum: int) -> Points:
    """Transform small Pillow geometry on CPU; no device scalar reads."""
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f"points must have shape (N, 2), with N >= {minimum}")
    if not np.isfinite(points).all() or not math.isfinite(angle):
        raise ValueError("Points and angle must be finite")
    points = points - points.mean(0)
    c, s = math.cos(angle), math.sin(angle)
    return points @ np.array([[c, -s], [s, c]], dtype=np.float32)


def _ellipse_coordinates(rx: float, ry: float, rotation: float) -> tuple[torch.Tensor, torch.Tensor]:
    _positive(rx=rx, ry=ry)
    if not math.isfinite(rotation):
        raise ValueError("rotation must be finite")
    cos_angle, sin_angle = math.cos(rotation), math.sin(rotation)
    extent_x = math.hypot(rx * cos_angle, ry * sin_angle)
    extent_y = math.hypot(rx * sin_angle, ry * cos_angle)
    half_width, half_height = int(math.ceil(extent_x)), int(math.ceil(extent_y))
    grid_y = torch.arange(
        -half_height, half_height + 1, dtype=torch.float32, device=device
    )[:, None]
    grid_x = torch.arange(
        -half_width, half_width + 1, dtype=torch.float32, device=device
    )[None, :]
    return (
        grid_x * cos_angle - grid_y * sin_angle,
        grid_x * sin_angle + grid_y * cos_angle,
    )


def _ring_mask(
    local_x: torch.Tensor, local_y: torch.Tensor,
    rx: float, ry: float, thickness: float,
) -> torch.Tensor:
    _positive(thickness=thickness)
    mask = (local_x / rx) ** 2 + (local_y / ry) ** 2 <= 1
    if thickness < min(rx, ry):
        mask &= (local_x / (rx - thickness)) ** 2 + (
            local_y / (ry - thickness)
        ) ** 2 >= 1
    return mask


def _mask_image(mask: torch.Tensor) -> torch.Tensor:
    return _crop_mask(mask.to(dtype=torch.float32)[..., None])


def _crop_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return the tight raster bounds, with one channel for RGB broadcasting."""
    occupied = mask[..., 0] != 0
    rows, columns = occupied.any(dim=1), occupied.any(dim=0)
    # Fixed-size reductions avoid nonzero's variable-sized device output.
    row_ids = torch.arange(mask.shape[0], device=mask.device)
    col_ids = torch.arange(mask.shape[1], device=mask.device)
    bounds = torch.stack((
        row_ids.masked_fill(~rows, mask.shape[0]).amin(),
        row_ids.masked_fill(~rows, -1).amax(),
        col_ids.masked_fill(~columns, mask.shape[1]).amin(),
        col_ids.masked_fill(~columns, -1).amax(),
    )).cpu().tolist()
    top, bottom, left, right = (int(value) for value in bounds)
    if bottom < top:
        return mask.new_zeros((1, 1, 1))
    return mask[top:bottom + 1, left:right + 1].clone()


def _draw_vertices(
    vertices: torch.Tensor | Points,
    line_width: int | None = None,
) -> torch.Tensor:
    output_device = device
    if isinstance(vertices, torch.Tensor):
        cpu_vertices = vertices.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        cpu_vertices = np.asarray(vertices, dtype=np.float32)

    padding = 0 if line_width is None else line_width
    lower = np.floor(cpu_vertices.min(axis=0)) - padding
    upper = np.ceil(cpu_vertices.max(axis=0)) + padding
    size = upper - lower + 1
    width, height = int(size[0]), int(size[1])

    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    points = [(float(x), float(y)) for x, y in cpu_vertices - lower]

    if line_width is None:
        draw.polygon(points, fill=255)
    else:
        draw.line(points, fill=255, width=line_width, joint="curve")

    bounds = image.getbbox()
    if bounds is None:
        return torch.zeros(
            (1, 1, 1), dtype=torch.float32, device=output_device
        )

    image = image.crop(bounds)
    width, height = image.size
    mask = torch.frombuffer(
        bytearray(image.tobytes()), dtype=torch.uint8
    ).reshape(height, width, 1)

    return mask.to(
        device=output_device, dtype=torch.float32
    ).mul_(NORMALIZE_RGB_VALUE)
