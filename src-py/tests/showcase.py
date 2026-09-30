"""Visual smoke tests: run main.py, or python -m generate_image.showcase --no-show."""

import argparse
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from ..generate_image import fills, shapes


def shape_cases():
    """Two contrasting examples per public shape, using a 256 x 256 canvas."""
    return [
        ("circle", "Round", dict(rx=75, ry=75, rotation=0)),
        ("circle", "Rotated ellipse", dict(rx=95, ry=35, rotation=np.pi / 4)),
        ("ring", "Thin circular ring", dict(rx=85, ry=85, thickness=3, rotation=0)),
        ("ring", "Thick rotated ellipse", dict(rx=95, ry=50, thickness=16, rotation=np.pi / 3)),
        ("pie_slice", "Filled sector", dict(rx=85, ry=85, thickness=85,
            start_angle=0, end_angle=2 * np.pi / 3, rotation=0)),
        ("pie_slice", "Elliptical arc crossing zero", dict(rx=95, ry=65, thickness=12,
            start_angle=5 * np.pi / 3, end_angle=np.pi / 3, rotation=np.pi / 4)),
        ("equilateral_triangle", "Upright", dict(radius=90, angle=0)),
        ("equilateral_triangle", "Small and rotated", dict(radius=48, angle=np.pi / 5, x=90, y=150)),
        ("isosceles_triangle", "Tall", dict(leg_a=75, leg_bc=145, angle=0)),
        ("isosceles_triangle", "Wide and rotated", dict(leg_a=150, leg_bc=85, angle=np.pi / 4)),
        ("scalene_triangle", "Right triangle", dict(leg_a=90, leg_b=120, leg_c=150, angle=0)),
        ("scalene_triangle", "Obtuse and rotated", dict(leg_a=160, leg_b=100, leg_c=75, angle=-np.pi / 5)),
        ("square", "Axis aligned", dict(radius=85, angle=0)),
        ("square", "Diamond", dict(radius=65, angle=np.pi / 4, x=100, y=145)),
        ("rectangle", "Wide", dict(side_a=170, side_b=65, angle=0)),
        ("rectangle", "Thin diagonal bar", dict(side_a=185, side_b=8, angle=np.pi / 3)),
        ("parallelogram", "Strong shear", dict(side_a=110, side_b=95, base_height=55, angle=0)),
        ("parallelogram", "Rotated", dict(side_a=95, side_b=90, base_height=85, angle=-np.pi / 4)),
        ("trapezoid", "Symmetric", dict(base_a=170, base_b=80, base_height=100, base_offset=45, angle=0)),
        ("trapezoid", "Offset and rotated", dict(base_a=100, base_b=150, base_height=80, base_offset=-65, angle=np.pi / 6)),
        ("star", "Deep notches", dict(inner_radius=28, outer_radius=95, angle=0)),
        ("star", "Broad rotated tips", dict(inner_radius=55, outer_radius=85, angle=np.pi / 5)),
        ("regular_polygon", "Pentagon", dict(radius=90, num_points=5, angle=0)),
        ("regular_polygon", "Rotated octagon", dict(radius=75, num_points=8, angle=np.pi / 8)),
        ("polygon", "Irregular convex boundary", dict(
            points=np.array([[0, 0], [120, -20], [150, 70], [70, 130], [-20, 80]]), angle=0)),
        ("polygon", "Concave arrow", dict(
            points=np.array([[-85, -25], [0, -25], [0, -65], [85, 0], [0, 65], [0, 25], [-85, 25]]),
            angle=np.pi / 6)),
        ("poly_line", "One-pixel segment", dict(
            points=np.array([[-90, 0], [90, 0]]), line_width=1, angle=np.pi / 7)),
        ("poly_line", "Thick open zigzag", dict(
            points=np.array([[-85, 55], [-45, -55], [0, 45], [45, -55], [85, 55]]),
            line_width=9, angle=-np.pi / 10)),
    ]


def show_shape_tests(show: bool = True) -> None:
    """Validate every example and optionally open one labeled window per case."""
    cases = shape_cases()
    counts = Counter(name for name, _, _ in cases)
    public_shapes = {
        name for name, value in vars(shapes).items()
        if not name.startswith("_") and callable(value)
        and getattr(value, "__module__", None) == shapes.__name__
    }
    assert set(counts) == public_shapes, "Update showcase cases to cover every shape"
    assert all(count >= 2 for count in counts.values())

    # Validate everything before opening windows, so a failure leaves no partial gallery.
    results = []
    for name, description, parameters in cases:
        arguments = dict(x=128, y=128, width=256, height=256)
        arguments.update(parameters)
        img = getattr(shapes, name)(**arguments)
        label = f"{name}: {description}"
        assert img.shape == (256, 256, 3), f"{label}: unexpected shape"
        assert img.dtype == np.float32, f"{label}: unexpected dtype"
        assert np.all(np.isfinite(img)), f"{label}: nonfinite pixels"
        assert img.min() == 0 and img.max() == 1, f"{label}: expected black and white pixels"
        assert np.all((img >= 0) & (img <= 1)), f"{label}: pixels outside [0, 1]"
        assert np.array_equal(img[:, :, 0], img[:, :, 1]), f"{label}: RGB channels differ"
        assert np.array_equal(img[:, :, 0], img[:, :, 2]), f"{label}: RGB channels differ"
        results.append((label, img, arguments))
        print(f"PASS {label}")

    print(f"Passed {len(results)} examples across {len(counts)} shapes.", flush=True)
    if not show:
        return

    # All windows open together; closing one does not prevent viewing the others.
    with plt.rc_context({"figure.max_open_warning": 0}):
        for index, (label, img, arguments) in enumerate(results, start=1):
            fig, ax = plt.subplots(num=f"{index:02d}/{len(results)} — {label}", figsize=(5, 5))
            ax.imshow(img, interpolation="nearest", origin="upper")
            ax.set_title(label.replace(": ", "\n"))
            ax.plot(arguments["x"], arguments["y"], "+", color="tomato", markersize=8)
            ax.set_xlabel("x (pixels); + marks requested center")
            ax.set_ylabel("y (pixels)")
            fig.tight_layout()
        plt.show()


def fill_cases():
    """Two contrasting examples per public fill, using a 256 x 256 canvas."""
    return [
        ("solid", "Warm orange", dict(color=[1.0, 0.3, 0.05])),
        ("solid", "Cool blue", dict(color=[0.05, 0.3, 0.9])),
        ("linear_gradient", "Horizontal black to white", dict(
            points=[[48, 128], [208, 128]], colors=[[0, 0, 0], [1, 1, 1]])),
        ("linear_gradient", "Diagonal three-channel blend", dict(
            points=[[64, 64], [192, 192]], colors=[[1, 0.1, 0.2], [0.1, 0.7, 1]])),
        ("radial_gradient", "Circular falloff", dict(
            center=[128, 128], radii=[90, 90], angle=0, stops=[0, 1],
            colors=[[1, 1, 1], [0.1, 0.1, 0.4]])),
        ("radial_gradient", "Rotated ellipse with three stops", dict(
            center=[110, 140], radii=[110, 45], angle=np.pi / 4, stops=[0, 0.4, 1],
            colors=[[1, 0.9, 0.1], [0.9, 0.1, 0.2], [0.1, 0, 0.3]])),
        ("stripes", "Equal vertical stripes", dict(
            stripe_widths=[16, 16], angle=0, offset=0, colors=[[0, 0, 0], [1, 1, 1]])),
        ("stripes", "Unequal diagonal stripes", dict(
            stripe_widths=[8, 20, 36], angle=np.pi / 4, offset=13,
            colors=[[1, 0.2, 0.1], [0.1, 0.8, 0.3], [0.1, 0.3, 1]])),
        ("checkerboard", "Square cells", dict(
            cell_size=[24, 24], angle=0, offset=[0, 0], colors=[[0, 0, 0], [1, 1, 1]])),
        ("checkerboard", "Rotated rectangular cells", dict(
            cell_size=[16, 40], angle=np.pi / 6, offset=[7, 11],
            colors=[[0.1, 0.2, 0.6], [1, 0.8, 0.2]])),
        ("sinusoidal_grating", "Fine vertical waves", dict(
            wavelength=12, angle=0, phase=0, colors=[[0, 0, 0], [1, 1, 1]])),
        ("sinusoidal_grating", "Broad diagonal color waves", dict(
            wavelength=64, angle=np.pi / 4, phase=np.pi / 2,
            colors=[[0.7, 0.1, 0.8], [0.1, 0.9, 0.5]])),
        ("noise", "Subtle neutral grain", dict(stddev=0.05, mean_color=[0.5, 0.5, 0.5], seed=17)),
        ("noise", "Strong warm grain", dict(stddev=0.3, mean_color=[0.8, 0.4, 0.2], seed=29)),
        ("smooth_noise", "Broad single-octave clouds", dict(
            scale=80, octaves=1, persistence=0.5, colors=[[0, 0, 0], [1, 1, 1]], seed=17)),
        ("smooth_noise", "Detailed multi-octave terrain", dict(
            scale=32, octaves=4, persistence=0.6,
            colors=[[0.05, 0.15, 0.4], [0.9, 0.85, 0.3]], seed=29)),
    ]


def show_fill_tests(show: bool = True) -> None:
    """Validate every fill and optionally display each one on the same square."""
    cases = fill_cases()
    counts = Counter(name for name, _, _ in cases)
    public_fills = {
        name for name, value in vars(fills).items()
        if not name.startswith("_") and callable(value)
        and getattr(value, "__module__", None) == fills.__name__
    }
    assert set(counts) == public_fills, "Update showcase cases to cover every fill"
    assert all(count >= 2 for count in counts.values())

    mask = shapes.square(radius=140, angle=0, x=128, y=128, width=256, height=256)
    results = []
    for name, description, parameters in cases:
        arguments = dict(width=256, height=256, **parameters)
        fill = getattr(fills, name)(**arguments)
        label = f"{name}: {description}"
        assert fill.shape == (256, 256, 3), f"{label}: unexpected shape"
        assert fill.dtype == np.float32, f"{label}: unexpected dtype"
        assert np.all(np.isfinite(fill)), f"{label}: nonfinite pixels"
        assert np.all((fill >= 0) & (fill <= 1)), f"{label}: pixels outside [0, 1]"
        if name == "solid":
            assert np.all(fill == np.asarray(parameters["color"], dtype=np.float32)), label
        else:
            assert np.any(np.ptp(fill, axis=(0, 1)) > 0), f"{label}: expected spatial variation"
        if name in {"noise", "smooth_noise"}:
            assert np.array_equal(fill, getattr(fills, name)(**arguments)), f"{label}: seed is not reproducible"
        img = fill * mask + np.float32(0.15) * (1 - mask)
        results.append((label, img))
        print(f"PASS {label}")

    print(f"Passed {len(results)} examples across {len(counts)} fills.", flush=True)
    if not show:
        return

    with plt.rc_context({"figure.max_open_warning": 0}):
        for index, (label, img) in enumerate(results, start=1):
            fig, ax = plt.subplots(num=f"Fill {index:02d}/{len(results)} — {label}", figsize=(5, 5))
            ax.imshow(img, interpolation="nearest", origin="upper")
            ax.set_title(label.replace(": ", "\n"))
            ax.set_xlabel("x (pixels)")
            ax.set_ylabel("y (pixels)")
            fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true", help="Run checks without opening windows")
    args = parser.parse_args()
    show_shape_tests(show=not args.no_show)
    show_fill_tests(show=not args.no_show)
