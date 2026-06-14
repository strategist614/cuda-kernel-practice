# CUDA path from conda
CUDA_PATH ?= /opt/conda

# RTX 3090 = 86
# RTX 4090 = 89
SM ?= 86

# Compiler
NVCC ?= $(CUDA_PATH)/bin/nvcc

# Target
TARGET := vectorAddMMAP

# Source files
HOST_SRC := vectorAddMMAP.cpp multidevicealloc_memmap.cpp
HOST_OBJ := $(HOST_SRC:.cpp=.o)

KERNEL_SRC := vectorAdd_kernel.cu
FATBIN := vectorAdd_kernel64.fatbin

# Include paths
INCLUDES := -I. -I$(CUDA_PATH)/include

# Link flags
LIBS := -L$(CUDA_PATH)/lib -L$(CUDA_PATH)/lib/stubs -lcuda

# Compile flags
NVCCFLAGS := -std=c++17 $(INCLUDES)

.PHONY: all clean run

all: $(TARGET)

$(FATBIN): $(KERNEL_SRC)
	$(NVCC) -fatbin -arch=sm_$(SM) $< -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(FATBIN) $(HOST_OBJ)
	$(NVCC) $(HOST_OBJ) $(LIBS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o $(FATBIN)