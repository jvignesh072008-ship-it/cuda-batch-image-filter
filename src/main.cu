// Copyright 2026. Batch GPU image processing: Gaussian blur + Sobel edges.
//
// Processes every binary PGM (P5) image in an input directory using custom
// CUDA kernels and writes the results to an output directory.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t err__ = (call);                                            \
    if (err__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                   cudaGetErrorString(err__), __FILE__, __LINE__);         \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

constexpr int kBlockDim = 16;
constexpr int kGaussSize = 5;

// 5x5 Gaussian kernel (sigma ~ 1.0), normalized to sum to 1.
__constant__ float kGaussWeights[kGaussSize * kGaussSize];

struct Image {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;
};

struct Options {
  std::string input_dir = "data/input";
  std::string output_dir = "data/output";
  std::string filter = "edges";  // blur | sobel | edges
  int max_images = 0;            // 0 = all
};

__device__ __forceinline__ int Clamp(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void GaussianBlurKernel(const uint8_t* in, uint8_t* out, int width,
                                   int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  float sum = 0.0f;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) {
      int xx = Clamp(x + dx, 0, width - 1);
      int yy = Clamp(y + dy, 0, height - 1);
      sum += kGaussWeights[(dy + 2) * kGaussSize + (dx + 2)] *
             static_cast<float>(in[yy * width + xx]);
    }
  }
  out[y * width + x] = static_cast<uint8_t>(fminf(255.0f, sum + 0.5f));
}

__global__ void SobelKernel(const uint8_t* in, uint8_t* out, int width,
                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  auto px = [&](int dx, int dy) -> int {
    return in[Clamp(y + dy, 0, height - 1) * width + Clamp(x + dx, 0, width - 1)];
  };
  int gx = -px(-1, -1) + px(1, -1) - 2 * px(-1, 0) + 2 * px(1, 0) -
           px(-1, 1) + px(1, 1);
  int gy = -px(-1, -1) - 2 * px(0, -1) - px(1, -1) + px(-1, 1) +
           2 * px(0, 1) + px(1, 1);
  float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
  out[y * width + x] = static_cast<uint8_t>(fminf(255.0f, mag));
}

bool ReadPgm(const fs::path& path, Image* img) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;
  std::string magic;
  file >> magic;
  if (magic != "P5") return false;
  auto skip_comments = [&file]() {
    file >> std::ws;
    while (file.peek() == '#') {
      std::string line;
      std::getline(file, line);
      file >> std::ws;
    }
  };
  int maxval = 0;
  skip_comments();
  file >> img->width;
  skip_comments();
  file >> img->height;
  skip_comments();
  file >> maxval;
  file.get();  // Single whitespace after header.
  if (maxval != 255 || img->width <= 0 || img->height <= 0) return false;
  img->pixels.resize(static_cast<size_t>(img->width) * img->height);
  file.read(reinterpret_cast<char*>(img->pixels.data()), img->pixels.size());
  return static_cast<size_t>(file.gcount()) == img->pixels.size();
}

bool WritePgm(const fs::path& path, const Image& img) {
  std::ofstream file(path, std::ios::binary);
  if (!file) return false;
  file << "P5\n" << img.width << " " << img.height << "\n255\n";
  file.write(reinterpret_cast<const char*>(img.pixels.data()),
             img.pixels.size());
  return static_cast<bool>(file);
}

void PrintUsage(const char* prog) {
  std::printf(
      "Usage: %s [--input_dir DIR] [--output_dir DIR]\n"
      "          [--filter blur|sobel|edges] [--max_images N]\n"
      "  --filter edges = Gaussian blur followed by Sobel (default)\n",
      prog);
}

bool ParseArgs(int argc, char** argv, Options* opts) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> const char* {
      return (i + 1 < argc) ? argv[++i] : nullptr;
    };
    const char* value = nullptr;
    if (arg == "--help" || arg == "-h") {
      return false;
    } else if (arg == "--input_dir" && (value = next())) {
      opts->input_dir = value;
    } else if (arg == "--output_dir" && (value = next())) {
      opts->output_dir = value;
    } else if (arg == "--filter" && (value = next())) {
      opts->filter = value;
    } else if (arg == "--max_images" && (value = next())) {
      opts->max_images = std::atoi(value);
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
      return false;
    }
  }
  return opts->filter == "blur" || opts->filter == "sobel" ||
         opts->filter == "edges";
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!ParseArgs(argc, argv, &opts)) {
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::fprintf(stderr, "No CUDA device found.\n");
    return EXIT_FAILURE;
  }
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s (compute %d.%d)\n", prop.name, prop.major, prop.minor);

  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(opts.input_dir)) {
    if (entry.path().extension() == ".pgm") files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  if (opts.max_images > 0 && files.size() > static_cast<size_t>(opts.max_images)) {
    files.resize(opts.max_images);
  }
  if (files.empty()) {
    std::fprintf(stderr, "No .pgm files found in %s\n", opts.input_dir.c_str());
    return EXIT_FAILURE;
  }
  fs::create_directories(opts.output_dir);

  // Build normalized Gaussian weights on the host.
  float weights[kGaussSize * kGaussSize];
  const float kernel_1d[kGaussSize] = {1.f, 4.f, 6.f, 4.f, 1.f};
  for (int y = 0; y < kGaussSize; ++y) {
    for (int x = 0; x < kGaussSize; ++x) {
      weights[y * kGaussSize + x] = kernel_1d[y] * kernel_1d[x] / 256.0f;
    }
  }
  CHECK_CUDA(cudaMemcpyToSymbol(kGaussWeights, weights, sizeof(weights)));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::printf("Processing %zu images, filter=%s\n", files.size(),
              opts.filter.c_str());
  double total_gpu_ms = 0.0;
  size_t total_pixels = 0;
  int processed = 0;

  for (const auto& path : files) {
    Image img;
    if (!ReadPgm(path, &img)) {
      std::fprintf(stderr, "Skipping unreadable file: %s\n", path.c_str());
      continue;
    }
    const size_t bytes = img.pixels.size();
    uint8_t *d_in = nullptr, *d_tmp = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_tmp, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    dim3 block(kBlockDim, kBlockDim);
    dim3 grid((img.width + kBlockDim - 1) / kBlockDim,
              (img.height + kBlockDim - 1) / kBlockDim);

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(d_in, img.pixels.data(), bytes,
                          cudaMemcpyHostToDevice));
    if (opts.filter == "blur") {
      GaussianBlurKernel<<<grid, block>>>(d_in, d_out, img.width, img.height);
    } else if (opts.filter == "sobel") {
      SobelKernel<<<grid, block>>>(d_in, d_out, img.width, img.height);
    } else {
      GaussianBlurKernel<<<grid, block>>>(d_in, d_tmp, img.width, img.height);
      SobelKernel<<<grid, block>>>(d_tmp, d_out, img.width, img.height);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpy(img.pixels.data(), d_out, bytes,
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    total_gpu_ms += ms;
    total_pixels += img.pixels.size();

    fs::path out_path = fs::path(opts.output_dir) /
                        (path.stem().string() + "_" + opts.filter + ".pgm");
    if (!WritePgm(out_path, img)) {
      std::fprintf(stderr, "Failed to write %s\n", out_path.c_str());
    }
    std::printf("[%4d/%zu] %s %dx%d  %.3f ms\n", ++processed, files.size(),
                path.filename().c_str(), img.width, img.height, ms);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_tmp));
    CHECK_CUDA(cudaFree(d_out));
  }

  std::printf("---- Summary ----\n");
  std::printf("Images processed : %d\n", processed);
  std::printf("Total pixels     : %zu\n", total_pixels);
  std::printf("Total GPU time   : %.3f ms (incl. H2D/D2H copies)\n",
              total_gpu_ms);
  if (processed > 0) {
    std::printf("Average per image: %.3f ms\n", total_gpu_ms / processed);
  }
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return EXIT_SUCCESS;
}
