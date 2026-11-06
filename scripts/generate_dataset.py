"""Generates a synthetic dataset of grayscale PGM images for the CUDA project.

Each image contains random shapes (rectangles, circles, lines) plus noise so
that blur and edge detection have visible effects.

Usage: python3 generate_dataset.py --count 200 --width 512 --height 512 \
           --output_dir data/input
"""

import argparse
import os

import numpy as np


def make_image(rng, width, height):
    img = np.full((height, width), rng.integers(30, 120), dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(rng.integers(4, 10)):
        shade = rng.integers(60, 255)
        kind = rng.integers(0, 2)
        if kind == 0:
            x0 = rng.integers(0, width - 20)
            y0 = rng.integers(0, height - 20)
            x1 = min(width, x0 + rng.integers(20, width // 2))
            y1 = min(height, y0 + rng.integers(20, height // 2))
            img[y0:y1, x0:x1] = shade
        else:
            cx = rng.integers(0, width)
            cy = rng.integers(0, height)
            r = rng.integers(10, min(width, height) // 4)
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = shade
    img += rng.normal(0, 8, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_pgm(path, img):
    height, width = img.shape
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (width, height))
        f.write(img.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="data/input")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.count):
        img = make_image(rng, args.width, args.height)
        write_pgm(os.path.join(args.output_dir, "img_%04d.pgm" % i), img)
    print("Wrote %d images (%dx%d) to %s" %
          (args.count, args.width, args.height, args.output_dir))


if __name__ == "__main__":
    main()
