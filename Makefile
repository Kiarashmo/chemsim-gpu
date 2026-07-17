# ChemSim-GPU build.
#
#   make            # build the CUDA engine (needs nvcc) -> build/chemsim
#   make test       # build + run the host-only CPU tests (needs only g++/clang++)
#   make clean
#
# On a machine with an NVIDIA GPU, `make` is all you need. On a machine without
# a GPU (e.g. a laptop), `make test` still builds and runs the correctness tests.

NVCC     ?= nvcc
CXX      ?= c++
# Set SM to your GPU (T4=75, RTX30xx=86, RTX40xx=89, A100=80). Default covers common cards.
SM       ?= 75
NVFLAGS  ?= -O3 -std=c++17 -Iinclude -arch=sm_$(SM) -lineinfo
CXXFLAGS ?= -O3 -std=c++17 -Iinclude -Wall -Wextra

BUILD    := build
ENGINE   := $(BUILD)/chemsim
TESTBIN  := $(BUILD)/test_cpu

CU_SRC   := src/main.cu src/kernels.cu
CPP_SRC  := src/fingerprint_io.cpp src/tanimoto_cpu.cpp

.PHONY: all test clean
all: $(ENGINE)

# The CUDA engine: nvcc compiles the .cu files and the shared .cpp core.
$(ENGINE): $(CU_SRC) $(CPP_SRC) | $(BUILD)
	$(NVCC) $(NVFLAGS) $(CU_SRC) $(CPP_SRC) -o $@

# CPU-only tests, no GPU required.
test: $(TESTBIN)
	./$(TESTBIN)

$(TESTBIN): tests/test_cpu.cpp $(CPP_SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_cpu.cpp $(CPP_SRC) -o $@

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
