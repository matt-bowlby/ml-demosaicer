import numpy as np
from PIL import Image, ImageDraw

NORMALIZE_RGB_VALUE = 0.00392156862745098 # 1 / 255

def circle(rx: float, ry: float, rotation: float) -> np.ndarray:
    """Draw a rotated ellipse in its tight raster bounding box."""
    local_x, local_y = _ellipse_coordinates(rx, ry, rotation)
    return _mask_image((local_x / rx)**2 + (local_y / ry)**2 <= 1)


def ring(rx: float, ry: float, thickness: float, rotation: float) -> np.ndarray:
    """Draw an elliptical ring; thickness is subtracted from both inner radii."""
    local_x, local_y = _ellipse_coordinates(rx, ry, rotation)
    return _mask_image(_ring_mask(local_x, local_y, rx, ry, thickness))

def pie_slice(rx: float, ry: float, thickness: float, start_angle: float, end_angle: float, rotation: float) -> np.ndarray:
    """Draw an elliptical ring sector, counterclockwise from start to end.

    Angles are radians from the local positive x-axis. A span of at least
    2*pi draws a full ring; thickness >= min(rx, ry) fills to the center.
    """
    local_x, local_y = _ellipse_coordinates(rx, ry, rotation)
    mask = _ring_mask(local_x, local_y, rx, ry, thickness)
    span = end_angle - start_angle
    if not np.isfinite(span):
        raise ValueError("Sector angles must be finite")
    if abs(span) < 2 * np.pi:
        if span == 0:
            mask[:] = False
        else:
            theta = np.arctan2(-local_y / ry, local_x / rx)
            angular_mask = (theta - start_angle) % (2 * np.pi) <= span % (2 * np.pi)
            angular_mask |= (local_x == 0) & (local_y == 0)
            mask &= angular_mask
    return _mask_image(mask)

def equilateral_triangle(radius: float, angle: float) -> np.ndarray:
    return regular_polygon(radius, 3, angle)

def isosceles_triangle(leg_a: float, leg_bc: float, angle: float) -> np.ndarray:
    half_base = leg_a / 2
    altitude = np.sqrt(leg_bc**2 - half_base**2)
    vertices = np.array([
        [0, -2 * altitude / 3],
        [-half_base, altitude / 3],
        [half_base, altitude / 3],
    ], dtype=np.float32)

    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32)
    vertices = vertices @ rotation

    return _draw_vertices(vertices)

def scalene_triangle(leg_a: float, leg_b: float, leg_c: float, angle: float) -> np.ndarray:
    apex_x = (leg_a**2 + leg_b**2 - leg_c**2) / (2 * leg_a)
    altitude = np.sqrt(max(0.0, leg_b**2 - apex_x**2))
    vertices = np.array([
        [apex_x, -altitude],
        [0, 0],
        [leg_a, 0],
    ], dtype=np.float32)
    vertices -= vertices.mean(axis=0)

    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32)
    vertices = vertices @ rotation

    return _draw_vertices(vertices)

def square(radius: float, angle: float) -> np.ndarray:
    """Draw a square with the given center-to-corner radius."""
    return regular_polygon(radius, 4, angle)

def rectangle(side_a: float, side_b: float, angle: float) -> np.ndarray:
    _positive(side_a=side_a, side_b=side_b)
    points = np.array([[0, 0], [side_a, 0], [side_a, -side_b], [0, -side_b]], dtype=np.float32)
    return polygon(points, angle)

def parallelogram(side_a: float, side_b: float, base_height: float, angle: float) -> np.ndarray:
    """Draw a parallelogram whose upper base is shifted to the right."""
    _positive(side_a=side_a, side_b=side_b, base_height=base_height)
    if base_height > side_b:
        raise ValueError("base_height must not exceed side_b")
    offset = np.sqrt(max(0.0, side_b**2 - base_height**2))
    return trapezoid(side_a, side_a, base_height, offset, angle)

def trapezoid(base_a: float, base_b: float, base_height: float, base_offset: float, angle: float) -> np.ndarray:
    """base_offset is the upper base's left endpoint relative to the lower one."""
    _positive(base_a=base_a, base_b=base_b, base_height=base_height)
    points = np.array([
        [0, 0], [base_a, 0],
        [base_offset + base_b, -base_height], [base_offset, -base_height],
    ], dtype=np.float32)
    return polygon(points, angle)

def star(inner_radius: float, outer_radius: float, points: int, angle: float) -> np.ndarray:
    """Draw a star with at least three points and an upward tip at zero rotation."""
    _positive(inner_radius=inner_radius, outer_radius=outer_radius)
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius")
    if not isinstance(points, (int, np.integer)) or points < 3:
        raise ValueError("points must be an integer of at least 3")
    indices = np.arange(2 * points, dtype=np.float32)
    theta = np.pi / 2 + indices * np.pi / points
    radii = np.where(indices % 2 == 0, outer_radius, inner_radius)
    vertices = np.column_stack((radii * np.cos(theta), -radii * np.sin(theta)))
    return polygon(vertices, angle)

def regular_polygon(radius: float, num_sides: int, angle: float) -> np.ndarray:
    """Draw a regular polygon with a horizontal bottom edge at zero rotation."""
    _positive(radius=radius)
    if not isinstance(num_sides, (int, np.integer)) or num_sides < 3:
        raise ValueError("num_points must be an integer of at least 3")
    theta = 3 * np.pi / 2 - np.pi / num_sides + np.arange(num_sides, dtype=np.float32) * 2 * np.pi / num_sides
    points = np.column_stack((radius * np.cos(theta), -radius * np.sin(theta)))
    vertices = np.round(_transform_points(points, angle, 3), 2)
    return _draw_vertices(vertices)

def polygon(points: np.ndarray, angle: float) -> np.ndarray:
    """Draw ordered boundary points in their tight raster bounding box."""
    vertices = _transform_points(points, angle, 3)
    return _draw_vertices(vertices)

def poly_line(points: np.ndarray, line_width: float, angle: float) -> np.ndarray:
    """Draw an open polyline; line_width is rounded to at least one pixel."""
    _positive(line_width=line_width)
    vertices = _transform_points(points, angle, 2)
    return _draw_vertices(vertices, max(1, round(line_width)))


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _transform_points(points, angle, minimum):
    vertices = np.asarray(points, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < minimum:
        raise ValueError(f"points must have shape (N, 2) with at least {minimum} vertices")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("points must be finite")
    vertices = vertices - vertices.mean(axis=0)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32)
    return vertices @ rotation


def _ellipse_coordinates(rx, ry, rotation):
    _positive(rx=rx, ry=ry)
    if not np.isfinite(rotation):
        raise ValueError("rotation must be finite")
    cos_angle, sin_angle = np.float32(np.cos(rotation)), np.float32(np.sin(rotation))
    extent_x = np.hypot(rx * cos_angle, ry * sin_angle)
    extent_y = np.hypot(rx * sin_angle, ry * cos_angle)
    half_width, half_height = int(np.ceil(extent_x)), int(np.ceil(extent_y))
    grid_y = np.arange(-half_height, half_height + 1, dtype=np.float32)[:, None]
    grid_x = np.arange(-half_width, half_width + 1, dtype=np.float32)[None, :]
    return grid_x * cos_angle - grid_y * sin_angle, grid_x * sin_angle + grid_y * cos_angle


def _ring_mask(local_x, local_y, rx, ry, thickness):
    _positive(thickness=thickness)
    mask = (local_x / rx)**2 + (local_y / ry)**2 <= 1
    if thickness < min(rx, ry):
        mask &= (local_x / (rx - thickness))**2 + (local_y / (ry - thickness))**2 >= 1
    return mask


def _mask_image(mask):
    return _crop_mask(mask.astype(np.float32)[..., None])


def _crop_mask(mask):
    """Return the tight raster bounds, with one channel for RGB broadcasting."""
    occupied = mask[..., 0] != 0
    rows = np.flatnonzero(occupied.any(axis=1))
    columns = np.flatnonzero(occupied.any(axis=0))
    if not len(rows):
        return np.zeros((1, 1, 1), dtype=np.float32)
    return mask[rows.min():rows.max() + 1, columns.min():columns.max() + 1].copy()


def _draw_vertices(vertices, line_width=None):
    padding = 0 if line_width is None else line_width
    lower = np.floor(vertices.min(axis=0)) - padding
    upper = np.ceil(vertices.max(axis=0)) + padding
    width, height = (upper - lower + 1).astype(int)
    image = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(image)
    points = [tuple(vertex) for vertex in vertices - lower]
    if line_width is None:
        draw.polygon(points, fill=255)
    else:
        draw.line(points, fill=255, width=line_width, joint="curve")
    return _crop_mask(np.asarray(image, dtype=np.float32)[..., None] * NORMALIZE_RGB_VALUE)
