# Please change the path locations to your own. We include the cuda and torch versions
# with which we achieved compilation

GXX ::= g++ -std=c++17
NVCC ::= /path/to/cuda-12.8/bin/nvcc

PROJECT_ROOT ::= .
SRC_DIR ::= $(PROJECT_ROOT)/src
INC_DIR ::= $(PROJECT_ROOT)/include
TEST_DIR ::= $(PROJECT_ROOT)/tests
BUILD_DIR ::= $(PROJECT_ROOT)/build

TORCH_LIB ::= /path/to/envs/<env_name>/lib/python3.10/site-packages/torch/lib
TORCH_INC ::= /path/to/envs/<env_name>/lib/python3.10/site-packages/torch/include

CUDA_LIB ::= /path/to/cuda-12.8/lib64
CUDA_INC ::= /path/to/cuda-12.8/include

PYTHON_INC ::= /path/to/envs/<env_name>/include/python3.10

INCLUDE ::= -I$(INC_DIR) -isystem $(TORCH_INC) -isystem $(TORCH_INC)/torch/csrc/api/include -isystem $(CUDA_INC) -isystem $(PYTHON_INC)

LIBPATHS ::= \
-L$(TORCH_LIB) -L$(CUDA_LIB)

LIBS ::= \
-lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -lcudart

NVCCFLAGS ::= \
-O3 \
-DTORCH_EXTENSION_NAME=conv2d_cuda \
-DTORCH_API_INCLUDE_EXTENSION_H \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr \
-gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 \
--compiler-options '-fPIC' \
-std=c++17 \

NVCCFLAGS_thorough ::= \
-O3 \
-DTORCH_EXTENSION_NAME=conv2d_cuda_thorough \
-DTORCH_API_INCLUDE_EXTENSION_H \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr \
-gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 \
--compiler-options '-fPIC' \
-std=c++17 \


## for debugging

# NVCCFLAGS ::= \
# -g -G -O0 \
# -DTORCH_EXTENSION_NAME=conv2d_cuda \
# -DTORCH_API_INCLUDE_EXTENSION_H \
# -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
# --expt-relaxed-constexpr \
# -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 \
# --compiler-options '-fPIC' \
# -std=c++17 \

## enable the additional flags when testing the cpu fallback

CFLAGS ::= \
-Wall -Wextra \
# -g -O0
# -fsanitize=address \
# -fsanitize=signed-integer-overflow \
# -fsanitize=float-cast-overflow \
# -fsanitize=undefined \
# -fsanitize=integer-divide-by-zero \
# -fsanitize=pointer-overflow \
# -fsanitize=null \
# -fsanitize=bounds-strict \
# -fsanitize=float-divide-by-zero \

all: build 

build: $(BUILD_DIR)/conv2d_cuda.so $(BUILD_DIR)/conv2d_thorough.so 

$(BUILD_DIR)/conv2d_cuda.so: $(SRC_DIR)/conv2d_cuda.cuda.o
	$(GXX) $(SRC_DIR)/conv2d_cuda.cuda.o -shared $(LIBPATHS) $(LIBS) -o $(BUILD_DIR)/conv2d_cuda.so

$(SRC_DIR)/conv2d_cuda.cuda.o: $(SRC_DIR)/conv2d_cuda.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $(SRC_DIR)/conv2d_cuda.cu -o $(SRC_DIR)/conv2d_cuda.cuda.o 

$(BUILD_DIR)/conv2d_thorough.so: $(SRC_DIR)/conv2d_thorough.cuda.o
	$(GXX) $(SRC_DIR)/conv2d_thorough.cuda.o -shared $(LIBPATHS) $(LIBS) -o $(BUILD_DIR)/conv2d_thorough.so

$(SRC_DIR)/conv2d_thorough.cuda.o: $(SRC_DIR)/conv2d_thorough.cu
	$(NVCC) $(NVCCFLAGS_thorough) $(INCLUDE) -c $(SRC_DIR)/conv2d_thorough.cu -o $(SRC_DIR)/conv2d_thorough.cuda.o 

## JIT compilation instructions which were used to develop this Makefile. 

# [1/2] /usr/local/cuda-12.8/bin/nvcc -MD -MF conv2d_cuda.cuda.o.d -DTORCH_EXTENSION_NAME=conv2d_cuda_v1 -DTORCH_API_INCLUDE_EXTENSION_H -isystem /gpu-data2/kfot/miniconda3/envs/torch213/lib/python3.10/site-packages/torch/include -isystem /gpu-data2/kfot/miniconda3/envs/torch213/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda-12.8/include -isystem /gpu-data2/kfot/miniconda3/envs/torch213/include/python3.10 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 --compiler-options '-fPIC' -std=c++17 -c /gpu-data2/kfot/MNNs_3/conv2d_cuda.cu -o conv2d_cuda.cuda.o 
# [2/2] c++ conv2d_cuda.cuda.o -shared -L/gpu-data2/kfot/miniconda3/envs/torch213/lib/python3.10/site-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -L/usr/local/cuda-12.8/lib64 -lcudart -o conv2d_cuda_v1.so
