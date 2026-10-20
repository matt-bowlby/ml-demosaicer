"""Run with: python -m generate_image.distribution_explorer"""

from math import lgamma

import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Slider
import numpy as np

def beta_parameters(peak: float, concentration: float) -> tuple[float, float]:
    """Convert a mode and concentration to Beta shape parameters.

    Higher concentration makes the distribution tighter around its peak.
    Zero concentration gives a uniform distribution (no unique peak).
    """
    if not np.isfinite(peak) or not 0 <= peak <= 1:
        raise ValueError("peak must be finite and within [0, 1]")
    if not np.isfinite(concentration) or concentration < 0:
        raise ValueError("concentration must be finite and nonnegative")
    return 1 + peak * concentration, 1 + (1 - peak) * concentration

def beta_density(x, a, b):
    """Evaluate the normalized density for shape parameters >= 1."""
    with np.errstate(divide="ignore"):
        left = (a - 1) * np.log(x) if a > 1 else np.zeros_like(x)
        right = (b - 1) * np.log1p(-x) if b > 1 else np.zeros_like(x)
    return np.exp(left + right + lgamma(a + b) - lgamma(a) - lgamma(b))


class DistributionExplorer:
    """Interactive preview of the Beta distribution sampled by rngd."""

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.subplots_adjust(left=0.12, right=0.95, bottom=0.40, top=0.90)
        self.fig.canvas.manager.set_window_title("Beta distribution explorer")
        self.ax.set(xlim=(0, 1), xlabel="Sample value", ylabel="Probability density",
                    title="Beta distribution on [0, 1]")
        self.ax.grid(alpha=0.2)
        self.x = np.linspace(0, 1, 2001)
        self.line, = self.ax.plot([], [], color="tab:blue", linewidth=2)
        self.shade = None
        self.summary = self.fig.text(0.12, 0.30, "", fontsize=11)
        self.code = self.fig.text(0.12, 0.25, "", family="monospace")
        self.peak = Slider(self.fig.add_axes((0.25, 0.18, 0.60, 0.03)),
                             "Peak", 0, 1, valinit=0.3, valstep=0.001, valfmt="%.3f")
        self.concentration = Slider(self.fig.add_axes((0.25, 0.12, 0.60, 0.03)),
                            "Concentration", 0, 50, valinit=4,
                            valstep=0.1, valfmt="%.1f")
        self.interval = RangeSlider(self.fig.add_axes((0.25, 0.06, 0.60, 0.03)),
                                    "Interval", 0, 1, valinit=(0.7, 0.9),
                                    valstep=0.001, valfmt="%.3f")
        for slider in (self.peak, self.concentration, self.interval):
            slider.on_changed(self.update)
        self.quad_x, self.quad_w = np.polynomial.legendre.leggauss(256)
        self.update()

    def update(self, _value=None):
        peak, concentration = self.peak.val, self.concentration.val
        low, high = self.interval.val
        a, b = beta_parameters(peak, concentration)
        density = beta_density(self.x, a, b)
        self.line.set_data(self.x, density)
        self.ax.set_ylim(0, density.max() * 1.1)
        if self.shade is not None:
            self.shade.remove()
        interval_x = np.linspace(low, high, 501)
        interval_y = beta_density(interval_x, a, b)
        self.shade = self.ax.fill_between(interval_x, interval_y, color="tab:blue", alpha=0.25)
        # Gaussian quadrature integrates the selected area without SciPy.
        nodes = low + (self.quad_x + 1) * (high - low) / 2
        probability = float(np.dot(self.quad_w, beta_density(nodes, a, b)) * (high - low) / 2)
        self.summary.set_text(
            f"Approx. chance within [{low:.3f}, {high:.3f}]: {probability:.1%}"
            "    |    Total area = 1"
        )
        self.code.set_text(f"rngd(peak={peak:.3f}, concentration={concentration:.1f})")
        self.fig.canvas.draw_idle()


def main():
    # Keep the sliders alive for as long as the window is open.
    explorer = DistributionExplorer()
    plt.show()
    return explorer


if __name__ == "__main__":
    main()
