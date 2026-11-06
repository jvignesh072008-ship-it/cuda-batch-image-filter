"""Converts a few input/output PGM pairs into PNGs for proof of execution."""

import argparse
import os

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/input")
    parser.add_argument("--output_dir", default="data/output")
    parser.add_argument("--preview_dir", default="artifacts/previews")
    parser.add_argument("--filter", default="edges")
    parser.add_argument("--num", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.preview_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".pgm"))
    for name in names[:args.num]:
        stem = name[:-4]
        before = Image.open(os.path.join(args.input_dir, name))
        after = Image.open(
            os.path.join(args.output_dir, "%s_%s.pgm" % (stem, args.filter)))
        before.save(os.path.join(args.preview_dir, "%s_before.png" % stem))
        after.save(os.path.join(args.preview_dir, "%s_after.png" % stem))
        side = np.hstack([np.array(before), np.array(after)])
        Image.fromarray(side).save(
            os.path.join(args.preview_dir, "%s_side_by_side.png" % stem))
    print("Wrote previews to %s" % args.preview_dir)


if __name__ == "__main__":
    main()
