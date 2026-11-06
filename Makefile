# Makefile for the CUDA batch image processing project.
#
#   make          - build bin/batch_image_filter
#   make run      - generate data (if needed) and run the full pipeline
#   make clean    - remove build products

NVCC ?= nvcc
ARCH ?= native
NVCCFLAGS ?= -O2 -std=c++17 -arch=$(ARCH)

TARGET := bin/batch_image_filter
SRC := src/main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./run.sh

clean:
	rm -rf bin

.PHONY: all run clean
