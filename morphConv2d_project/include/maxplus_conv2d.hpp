#pragma once
#include <bits/stdc++.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

using c10_float = c10::impl::ScalarTypeToCPPType<c10::ScalarType::Float>::type; 
using c10_double = c10::impl::ScalarTypeToCPPType<c10::ScalarType::Double>::type; 

#define THREADS_PER_BLOCK 16
// #define STRIDE 5
#define STRIDE 1

// #define BLOCKSIZE 32
// #define STRIDE 1
// #define SHARED_SIZE 16

void maxplus_conv2d_forward_cpu_fallback(
    const c10_float*, const c10_float*, const c10_float*, const bool*,
    int*, 
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, 
    c10_float*, ssize_t*, ssize_t*, bool*
); 

void maxplus_conv2d_forward_cpu_fallback(
    const c10_double*, const c10_double*, const c10_double*, const bool*,
    int*, 
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, 
    c10_double*, ssize_t*, ssize_t*, bool*
); 

void maxplus_conv2d_backward_cpu_fallback(
    const c10_float*, const ssize_t*, const ssize_t*, const bool*, 
    int*, 
    int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, 
    c10_float*, c10_float*, c10_float*
); 

void maxplus_conv2d_backward_cpu_fallback(
    const c10_double*, const ssize_t*, const ssize_t*, const bool*, 
    int*, 
    int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, 
    c10_double*, c10_double*, c10_double*
); 

__global__ void maxplus_conv2d_forward_gpu_kernel(
    const c10_float*, const c10_float*, const c10_float*, const bool*,
    int*, 
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, 
    c10_float*, ssize_t*, ssize_t*, bool*
); 

__global__ void maxplus_conv2d_forward_gpu_kernel(
    const c10_double*, const c10_double*, const c10_double*, const bool*,
    int*, 
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, 
    c10_double*, ssize_t*, ssize_t*, bool*
); 

__global__ void maxplus_conv2d_backward_gpu_kernel(
    const c10_float*, const ssize_t*, const ssize_t*, const bool*, 
    int*, 
    int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, 
    c10_float*, c10_float*, c10_float*
); 

__global__ void maxplus_conv2d_backward_gpu_kernel(
    const c10_double*, const ssize_t*, const ssize_t*, const bool*, 
    int*, 
    int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, 
    c10_double*, c10_double*, c10_double*
); 

std::vector<torch::Tensor> maxplus_conv2d_forward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t
); 

std::vector<torch::Tensor> maxplus_conv2d_backward(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
    int64_t, int64_t, int64_t, int64_t, int64_t, int64_t
); 