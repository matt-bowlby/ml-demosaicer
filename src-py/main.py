import numpy as np
import matplotlib.pyplot as plt

# from tests.showcase import show_fill_tests, show_shape_tests
from generate_image import generate_image

def main():
    rng = np.random.default_rng(42)

    _, axes = plt.subplots(3, 4, figsize=(12, 6))

    for ax in axes.flat:
        image = generate_image(rng, 400, 250)
        ax.imshow(image)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
