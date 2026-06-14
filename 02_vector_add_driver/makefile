CUDA_PATH   ?= /opt/conda
CUDA_TARGET ?= /opt/conda/targets/x86_64-linux

NVCC := $(CUDA_PATH)/bin/nvcc
CXX  := g++

TARGET     := vectorAddDrv
HOST_SRC   := vectorAddDrv.cpp
KERNEL_SRC := vectorAdd.cu
FATBIN     := vectorAdd_kernel.fatbin

# RTX 3090 = sm_86
CUDA_ARCH ?= 86

CXXFLAGS := -std=c++11 \
            -I. \
            -I$(CUDA_TARGET)/include

LDFLAGS := -L$(CUDA_TARGET)/lib \
           -L$(CUDA_TARGET)/lib/stubs \
           -L/usr/lib/x86_64-linux-gnu \
           -lcuda

NVCCFLAGS := -fatbin \
             -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)

.PHONY: all run clean info

all: run

info:
	@echo "CUDA_PATH   = $(CUDA_PATH)"
	@echo "CUDA_TARGET = $(CUDA_TARGET)"
	@echo "NVCC        = $(NVCC)"
	@echo "CUDA_ARCH   = sm_$(CUDA_ARCH)"
	@echo "CXXFLAGS    = $(CXXFLAGS)"
	@echo "LDFLAGS     = $(LDFLAGS)"
	@echo "NVCCFLAGS   = $(NVCCFLAGS)"

$(FATBIN): $(KERNEL_SRC)
	$(NVCC) $(NVCCFLAGS) $(KERNEL_SRC) -o $(FATBIN)

$(TARGET): $(HOST_SRC) $(FATBIN)
	$(CXX) $(HOST_SRC) -o $(TARGET) $(CXXFLAGS) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(FATBIN)