# Compiler
NVCC ?= nvcc

# Target executable
TARGET := vectorAdd_nvrtc

# Source files
SRC := vectorAdd.cpp
KERNEL := vectorAdd_kernel.cu

# Include path
INCLUDES := -I.

# Compile flags
NVCCFLAGS := -std=c++14 -O2

# Libraries
LIBS := -lnvrtc -lcuda

# Default target
all: $(TARGET)

$(TARGET): $(SRC) $(KERNEL) nvrtc_helper.h helper_cuda_drvapi.h helper_functions.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SRC) $(LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean