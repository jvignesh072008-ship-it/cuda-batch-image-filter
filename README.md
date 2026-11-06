# Batch GPU Image Filtering with CUDA (Gaussian Blur + Sobel Edge Detection)

Project for the *CUDA at Scale for the Enterprise* course (independent project).

## Overview

This program processes a large batch of grayscale images (hundreds of small
images or tens of large ones) on the GPU using custom CUDA kernels:

* **Gaussian blur**: 5x5 separable-weight kernel (1-4-6-4-1), stored in
  constant memory.
* **Sobel edge detection**: 3x3 horizontal and vertical gradients combined into
  a gradient magnitude.
* **edges** (default): blur first (noise suppression), then Sobel, with the
  intermediate image staying in GPU memory.

Images are read as binary PGM (P5, 8-bit). Each image is copied to the GPU,
filtered, copied back and written to the output directory. Per-image GPU time
(including host/device transfers) is measured with CUDA events and a summary is
printed at the end.

## Repository layout

```
.
├── Makefile                    # build (make), run (make run), clean
├── run.sh                      # build + generate data + run + save logs/previews
├── src/main.cu                 # CUDA kernels and command line program
├── scripts/generate_dataset.py # synthetic PGM dataset generator
├── scripts/make_previews.py    # before/after PNG previews
├── artifacts/                  # execution log and previews (created by run.sh)
└── data/{input,output}/        # images (created at run time)
```

## Requirements

* NVIDIA GPU and CUDA Toolkit (`nvcc`), C++17 compiler
* Python 3 with `numpy` and `Pillow` (only for dataset generation / previews)

## Build

```
make                 # builds bin/batch_image_filter
make ARCH=sm_75      # optionally choose a specific GPU architecture
```

If your `nvcc` does not support `-arch=native`, pass `ARCH=sm_XX` for your GPU.

## Run

Quick start (generates 200 images of 512x512, runs, saves proof of execution):

```
./run.sh                    # defaults: 200 images, 512x512, filter=edges
./run.sh 500 256 256 blur   # COUNT WIDTH HEIGHT FILTER
```

Manual use:

```
python3 scripts/generate_dataset.py --count 200 --width 512 --height 512 --output_dir data/input
./bin/batch_image_filter --input_dir data/input --output_dir data/output --filter edges
```

### Command line arguments

| Argument       | Default       | Description                                   |
|----------------|---------------|-----------------------------------------------|
| `--input_dir`  | `data/input`  | Directory containing `.pgm` images            |
| `--output_dir` | `data/output` | Where filtered images are written             |
| `--filter`     | `edges`       | `blur`, `sobel` or `edges` (blur then Sobel)  |
| `--max_images` | `0` (all)     | Limit the number of images processed          |
| `--help`       |               | Print usage                                   |

## Using other data

Any 8-bit binary PGM images work. Convert other datasets (e.g. the USC SIPI
database) with ImageMagick: `mogrify -format pgm -colorspace Gray *.tif`.

## Proof of execution

`run.sh` writes `artifacts/execution_log.txt` (GPU name, per-image timings,
summary) and PNG before/after images in `artifacts/previews/`. Zip or tar these
for submission: `tar czf proof_of_execution.tar.gz artifacts`.

## Notes

* Code follows the Google C++ Style Guide (naming, comments, namespaces,
  2-space indent, 80-column lines where practical).
* Boundary pixels use clamp-to-edge handling.
