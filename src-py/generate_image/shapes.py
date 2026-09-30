import numpy as np
from PIL import Image, ImageDraw

NORMALIZE_RGB_VALUE = 0.00392156862745098 # 1 / 255

def circle(rx: float, ry: float, x: float, y: float, rotation: float, width: int, height: int) -> np.ndarray:
    """Draw an ellipse rotated about (x, y), clipping only the final shape."""
    local_x, local_y = _ellipse_coordinates(rx, ry, rotation, x, y, width, height)
    return _mask_image((local_x / rx)**2 + (local_y / ry)**2 <= 1)


def ring(rx: float, ry: float, thickness: float, rotation: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    """Draw an elliptical ring; thickness is subtracted from both inner radii."""
    local_x, local_y = _ellipse_coordinates(rx, ry, rotation, x, y, width, height)
    return _mask_image(_ring_mask(local_x, local_y, rx, ry, thickness))

def pie_slice(rx: float, ry: float, thickness: float, start_angle: float, end_angle: float, rotation: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    """Draw an elliptical ring sector, counterclockwise from start to end.

    Angles are radians from the local positive x-axis. A span of at least
    2*pi draws a full ring; thickness >= min(rx, ry) fills to the center.
    """
    local_x, local_y = _ellipse_coordinates(rx, ry, rotation, x, y, width, height)
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

def equilateral_triangle(radius: float, angle: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    return regular_polygon(radius, 3, angle, x, y, width, height)

def isosceles_triangle(leg_a: float, leg_bc: float, angle: float, x: float, y: float, width: int, height: int) -> np.ndarray:#
    half_base = leg_a / 2
    altitude = np.sqrt(leg_bc**2 - half_base**2)
    vertices = np.array([
        [0, -2 * altitude / 3],
        [-half_base, altitude / 3],
        [half_base, altitude / 3],
    ])

    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    vertices = vertices @ rotation + np.array([x, y])

    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    draw.polygon([tuple(vertex) for vertex in vertices], fill="white")

    return np.asarray(image, dtype=np.float32) * NORMALIZE_RGB_VALUE

def scalene_triangle(leg_a: float, leg_b: float, leg_c: float, angle: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    apex_x = (leg_a**2 + leg_b**2 - leg_c**2) / (2 * leg_a)
    altitude = np.sqrt(max(0.0, leg_b**2 - apex_x**2))
    vertices = np.array([
        [apex_x, -altitude],
        [0, 0],
        [leg_a, 0],
    ])
    vertices -= vertices.mean(axis=0)

    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    vertices = vertices @ rotation + np.array([x, y])

    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    draw.polygon([tuple(vertex) for vertex in vertices], fill="white")

    return np.asarray(image, dtype=np.float32) * NORMALIZE_RGB_VALUE

def square(radius: float, angle: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    """Draw a square with the given center-to-corner radius."""
    return regular_polygon(radius, 4, angle, x, y, width, height)

def rectangle(side_a: float, side_b: float, angle: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    _positive(side_a=side_a, side_b=side_b)
    points = np.array([[0, 0], [side_a, 0], [side_a, -side_b], [0, -side_b]])
    return polygon(points, angle, x, y, width, height)

def parallelogram(side_a: float, side_b: float, base_height: float, angle: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    """Draw a parallelogram whose upper base is shifted to the right."""
    _positive(side_a=side_a, side_b=side_b, base_height=base_height)
    if base_height > side_b:
        raise ValueError("base_height must not exceed side_b")
    offset = np.sqrt(max(0.0, side_b**2 - base_height**2))
    return trapezoid(side_a, side_a, base_height, offset, angle, x, y, width, height)

def trapezoid(base_a: float, base_b: float, base_height: float, base_offset: float, angle: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    """base_offset is the upper base's left endpoint relative to the lower one."""
    _positive(base_a=base_a, base_b=base_b, base_height=base_height)
    points = np.array([
        [0, 0], [base_a, 0],
        [base_offset + base_b, -base_height], [base_offset, -base_height],
    ])
    return polygon(points, angle, x, y, width, height)

def star(inner_radius: float, outer_radius: float, points: int, angle: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    """Draw a star with at least three points and an upward tip at zero rotation."""
    _positive(inner_radius=inner_radius, outer_radius=outer_radius)
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius")
    if not isinstance(points, (int, np.integer)) or points < 3:
        raise ValueError("points must be an integer of at least 3")
    indices = np.arange(2 * points)
    theta = np.pi / 2 + indices * np.pi / points
    radii = np.where(indices % 2 == 0, outer_radius, inner_radius)
    vertices = np.column_stack((radii * np.cos(theta), -radii * np.sin(theta)))
    return polygon(vertices, angle, x, y, width, height)

def regular_polygon(radius: float, num_sides: int, angle: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    """Draw a regular polygon with a horizontal bottom edge at zero rotation."""
    _positive(radius=radius)
    if not isinstance(num_sides, (int, np.integer)) or num_sides < 3:
        raise ValueError("num_points must be an integer of at least 3")
    theta = 3 * np.pi / 2 - np.pi / num_sides + np.arange(num_sides) * 2 * np.pi / num_sides
    points = np.column_stack((radius * np.cos(theta), -radius * np.sin(theta)))
    vertices = np.round(_transform_points(points, angle, x, y, 3), 2)
    image = Image.new("RGB", (width, height), "black")
    ImageDraw.Draw(image).polygon([tuple(vertex) for vertex in vertices], fill="white")
    return np.asarray(image, dtype=np.float32) * NORMALIZE_RGB_VALUE

def polygon(points: np.ndarray, angle: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    """Draw ordered boundary points, centering their arithmetic mean on (x, y)."""
    vertices = _transform_points(points, angle, x, y, 3)
    image = Image.new("RGB", (width, height), "black")
    ImageDraw.Draw(image).polygon([tuple(vertex) for vertex in vertices], fill="white")
    return np.asarray(image, dtype=np.float32) * NORMALIZE_RGB_VALUE

def poly_line(points: np.ndarray, line_width: float, angle: float, x: float, y: float, width: int, height: int) -> np.ndarray:
    """Draw an open polyline; line_width is rounded to at least one pixel."""
    _positive(line_width=line_width)
    vertices = _transform_points(points, angle, x, y, 2)
    image = Image.new("RGB", (width, height), "black")
    ImageDraw.Draw(image).line(
        [tuple(vertex) for vertex in vertices], fill="white",
        width=max(1, round(line_width)), joint="curve",
    )
    return np.asarray(image, dtype=np.float32) * NORMALIZE_RGB_VALUE


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


def _transform_points(points, angle, x, y, minimum):
    vertices = np.asarray(points, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < minimum:
        raise ValueError(f"points must have shape (N, 2) with at least {minimum} vertices")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("points must be finite")
    vertices = vertices - vertices.mean(axis=0)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    return vertices @ rotation + np.array([x, y])


def _ellipse_coordinates(rx, ry, rotation, x, y, width, height):
    _positive(rx=rx, ry=ry)
    grid_y, grid_x = np.ogrid[:height, :width]
    dx, dy = grid_x - x, grid_y - y
    cos_angle, sin_angle = np.cos(rotation), np.sin(rotation)
    return dx * cos_angle - dy * sin_angle, dx * sin_angle + dy * cos_angle


def _ring_mask(local_x, local_y, rx, ry, thickness):
    _positive(thickness=thickness)
    mask = (local_x / rx)**2 + (local_y / ry)**2 <= 1
    if thickness < min(rx, ry):
        mask &= (local_x / (rx - thickness))**2 + (local_y / (ry - thickness))**2 >= 1
    return mask


def _mask_image(mask):
    return np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
