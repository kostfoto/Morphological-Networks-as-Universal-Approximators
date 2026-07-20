#include <bits/stdc++.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "maxplus_conv2d.hpp"


#define myassert(expr) if(!(expr)) { *error_flag = 1; printf("Host error\n"); throw(-1); }
#define mydeviceassert(expr) if(!(expr)) { atomicExch(error_flag, 1); printf("Kernel Error\n"); return; }

// Constraint 1: C_in, C_out, K, stride, padding, resol_j = H_j * W_j should all fit in int64_t. 
// Constraint 2: batch size B should fit in int64_t
// Constraint 3: int64_t should be 8-bytes. 
// Constraint 4: all array indices should fit in ssize_t 

// fixed dilation = 1 and groups = 1. Other options not-supported. 
void maxplus_conv2d_forward_cpu_fallback(
    const c10_float* input, const c10_float* weight, const c10_float* bias, const bool* dead_input,
    int* error_flag, 
    int64_t K, int64_t stride, int64_t padding, int64_t B, int64_t C_in, int64_t C_out,
    int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out, 
    c10_float* output, ssize_t* argmax_input, ssize_t* argmax_weight, bool* dead_output) {

        for(int64_t b = 0; b < B; b++) {
            for(int64_t oc = 0; oc < C_out; oc++) {
                for(int64_t i = 0; i < H_out; i++) {
                    for(int64_t j = 0; j < W_out; j++) {
                        // This never overflows by assumption
                        ssize_t out_idx = (((ssize_t)b * C_out + oc) * H_out + i) * W_out + j; 
                        c10_float res = -(c10_float)INFINITY; 
                        ssize_t argin = -2;  
                        ssize_t argw = -2; 
                        bool is_dead = true; 
                        for(int64_t ic = 0; ic < C_in; ic++) {
                            for(int64_t di = 0; di < K; di++) {
                                for(int64_t dj = 0; dj < K; dj++) {
                                    myassert((ssize_t)i * stride >= INT64_MIN + padding);
                                    myassert((ssize_t)j * stride >= INT64_MIN + padding);
                                    // This should never overflow (int32 * int32 -> int64)
                                    auto ii = (ssize_t)i * stride + di - padding; 
                                    auto jj = (ssize_t)j * stride + dj - padding; 

                                    if(ii < 0 || jj < 0 || 
                                        ii >= H_in || jj >= W_in) continue; 
                                    
                                    auto input_idx = (((ssize_t)b * C_in + ic) * H_in + ii) * W_in + jj; 
                                    auto weight_idx = (((ssize_t)oc * C_in + ic) * K + di) * K + dj; 

                                    myassert(0 <= input_idx && input_idx < (ssize_t)B * C_in * H_in * W_in); 
                                    myassert(0 <= weight_idx && weight_idx < (ssize_t)C_out * C_in * K * K); 
                                    if(dead_input != nullptr && dead_input[input_idx]) continue;
                                    // printf("hello1\n");
                                    // printf("%ld, %ld\n", input_idx, weight_idx); 
                                    auto val = input[input_idx] + weight[weight_idx]; 
                                    // printf("hello2\n"); 
                                    if(val > res) {
                                        res = val; 
                                        argin = input_idx; 
                                        argw = weight_idx; 
                                        is_dead = false;
                                    }
                                }
                            }
                        }
                        myassert(0 <= oc && oc < C_out);
                        if(bias != nullptr && bias[oc] > res) {
                            res = bias[oc];
                            argin = -1;   
                            argw = -1;
                            is_dead = false;
                        }

                        myassert(0 <= out_idx && out_idx < (((ssize_t)B * C_out * H_out * W_out))); 
                        output[out_idx] = res; 
                        argmax_input[out_idx] = argin; 
                        argmax_weight[out_idx] = argw; 
                        if(dead_input != nullptr) dead_output[out_idx] = is_dead; 
                    }
                }
            }
        }
    }

void maxplus_conv2d_forward_cpu_fallback(
    const c10_double* input, const c10_double* weight, const c10_double* bias, const bool* dead_input,
    int* error_flag, 
    int64_t K, int64_t stride, int64_t padding, int64_t B, int64_t C_in, int64_t C_out,
    int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out, 
    c10_double* output, ssize_t* argmax_input, ssize_t* argmax_weight, bool* dead_output) {

        for(int64_t b = 0; b < B; b++) {
            for(int64_t oc = 0; oc < C_out; oc++) {
                for(int64_t i = 0; i < H_out; i++) {
                    for(int64_t j = 0; j < W_out; j++) {
                        // This never overflows by assumption
                        ssize_t out_idx = (((ssize_t)b * C_out + oc) * H_out + i) * W_out + j; 
                        c10_double res = -(c10_double)INFINITY; 
                        ssize_t argin = -2;  
                        ssize_t argw = -2; 
                        bool is_dead = true; 
                        for(int64_t ic = 0; ic < C_in; ic++) {
                            for(int64_t di = 0; di < K; di++) {
                                for(int64_t dj = 0; dj < K; dj++) {
                                    myassert((ssize_t)i * stride >= INT64_MIN + padding);
                                    myassert((ssize_t)j * stride >= INT64_MIN + padding);
                                    // This should never overflow (int32 * int32 -> int64)
                                    auto ii = (ssize_t)i * stride + di - padding; 
                                    auto jj = (ssize_t)j * stride + dj - padding; 

                                    if(ii < 0 || jj < 0 || 
                                        ii >= H_in || jj >= W_in) continue; 
                                    
                                    auto input_idx = (((ssize_t)b * C_in + ic) * H_in + ii) * W_in + jj; 
                                    auto weight_idx = (((ssize_t)oc * C_in + ic) * K + di) * K + dj; 

                                    myassert(0 <= input_idx && input_idx < (ssize_t)B * C_in * H_in * W_in); 
                                    myassert(0 <= weight_idx && weight_idx < (ssize_t)C_out * C_in * K * K); 
                                    if(dead_input != nullptr && dead_input[input_idx]) continue;
                                    // printf("hello1\n");
                                    // printf("%ld, %ld\n", input_idx, weight_idx); 
                                    auto val = input[input_idx] + weight[weight_idx]; 
                                    // printf("hello2\n"); 
                                    if(val > res) {
                                        res = val; 
                                        argin = input_idx; 
                                        argw = weight_idx; 
                                        is_dead = false;
                                    }
                                }
                            }
                        }
                        myassert(0 <= oc && oc < C_out);
                        if(bias != nullptr && bias[oc] > res) {
                            res = bias[oc];
                            argin = -1;   
                            argw = -1;
                            is_dead = false;
                        }

                        myassert(0 <= out_idx && out_idx < (((ssize_t)B * C_out * H_out * W_out))); 
                        output[out_idx] = res; 
                        argmax_input[out_idx] = argin; 
                        argmax_weight[out_idx] = argw; 
                        if(dead_input != nullptr) dead_output[out_idx] = is_dead; 
                    }
                }
            }
        }
    }

void maxplus_conv2d_backward_cpu_fallback(
    const c10_float* grad_output, const ssize_t* argmax_input, const ssize_t* argmax_weight, const bool* dead_output,
    int* error_flag, 
    int64_t K, int64_t B, int64_t C_in, int64_t C_out,
    int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out, 
    c10_float* grad_input, c10_float* grad_weight, c10_float* grad_bias = nullptr) {

        for(int64_t b = 0; b < B; b++) {
            for(int64_t oc = 0; oc < C_out; oc++) {
                for(int64_t i = 0; i < H_out; i++) {
                    for(int64_t j = 0; j < W_out; j++) {
                        auto out_idx = (((ssize_t)b * C_out + oc) * H_out + i) * W_out + j; 
                        myassert(0 <= out_idx && out_idx < (((ssize_t)B * C_out * H_out * W_out)));
                        auto go = grad_output[out_idx]; 

                        if(dead_output != nullptr && dead_output[out_idx]) continue;
                        auto argin = argmax_input[out_idx]; 
                        auto argw = argmax_weight[out_idx]; 
                        
                        myassert(argin > -2 && argw > -2); 

                        if(argin != -1 && argw != -1) {
                            myassert(argin < (ssize_t)B * C_in * H_in * W_in && argw < (ssize_t)C_out * C_in * K * K);
                            grad_input[argin] += go; 
                            grad_weight[argw] += go; 
                        } else if(argin == -1 && argw == -1 && grad_bias != nullptr) {
                            myassert(oc >= 0 && oc < C_out);
                            grad_bias[oc] += go; 
                        } else myassert(false); 
                    }
                }
            }
        }
    }

void maxplus_conv2d_backward_cpu_fallback(
    const c10_double* grad_output, const ssize_t* argmax_input, const ssize_t* argmax_weight, const bool* dead_output, 
    int* error_flag, 
    int64_t K, int64_t B, int64_t C_in, int64_t C_out,
    int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out, 
    c10_double* grad_input, c10_double* grad_weight, c10_double* grad_bias = nullptr) {

        for(int64_t b = 0; b < B; b++) {
            for(int64_t oc = 0; oc < C_out; oc++) {
                for(int64_t i = 0; i < H_out; i++) {
                    for(int64_t j = 0; j < W_out; j++) {
                        auto out_idx = (((ssize_t)b * C_out + oc) * H_out + i) * W_out + j; 
                        myassert(0 <= out_idx && out_idx < (((ssize_t)B * C_out * H_out * W_out)));
                        auto go = grad_output[out_idx]; 

                        if(dead_output != nullptr && dead_output[out_idx]) continue;
                        auto argin = argmax_input[out_idx]; 
                        auto argw = argmax_weight[out_idx]; 
                        
                        myassert(argin > -2 && argw > -2); 

                        if(argin != -1 && argw != -1) {
                            myassert(argin < (ssize_t)B * C_in * H_in * W_in && argw < (ssize_t)C_out * C_in * K * K);
                            grad_input[argin] += go; 
                            grad_weight[argw] += go; 
                        } else if(argin == -1 && argw == -1 && grad_bias != nullptr) {
                            assert(oc < C_out);
                            grad_bias[oc] += go; 
                        } else myassert(false); 
                    }
                }
            }
        }
    }

// Correctness constraint: The kernel should always be run from within its corresponding entry point. 
__global__ void maxplus_conv2d_forward_gpu_kernel(
    const c10_float* input, const c10_float* weight, const c10_float* bias, const bool* dead_input,
    int* error_flag, 
    int64_t K, int64_t stride, int64_t padding, int64_t B, int64_t C_in, int64_t C_out,
    int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out, 
    c10_float* output, ssize_t* argmax_input, ssize_t* argmax_weight, bool* dead_output) {

        if(*error_flag) return; 

        __syncthreads();
        for(ssize_t i = threadIdx.y + blockDim.y * blockIdx.y; i < (ssize_t)H_out; i += gridDim.y * blockDim.y) {
            for(ssize_t j = threadIdx.z + blockDim.z * blockIdx.z; j < (ssize_t)W_out; j += gridDim.z * blockDim.z) {
                for(ssize_t out_idx_bc = threadIdx.x + blockDim.x * blockIdx.x; out_idx_bc < (ssize_t)B * C_out; out_idx_bc += gridDim.x * blockDim.x) {
                    auto out_idx = (out_idx_bc * H_out + i) * W_out + j; 
                    int64_t oc = out_idx_bc % C_out; 
                    int64_t b = out_idx_bc / C_out; 
                    // printf("%ld, %ld, %ld, %ld\n", b, oc, i, j); 
                    // printf("Hello 1: %ld, %ld, %ld, %ld\n", b, oc, i, j); 
                    mydeviceassert(b >= 0 && oc >= 0 && i >= 0 && j >= 0 &&
                            b < B && oc < C_out && i < H_out && j < W_out); 
                    // Same correctness guarrantee as the serial one, from this point onward everything should
                    // work as is. 
                    c10_float res = -(c10_float)INFINITY; 
                    ssize_t argin = -2;  
                    ssize_t argw = -2; 
                    bool is_dead = true; 
                    for(int64_t ic = 0; ic < C_in; ic++) {
                        for(int64_t di = 0; di < K; di++) {
                            for(int64_t dj = 0; dj < K; dj++) {
                                // printf("%ld, %ld, %ld\n", i, stride, padding); 
                                mydeviceassert((ssize_t)i * stride >= INT64_MIN + padding);
                                mydeviceassert((ssize_t)j * stride >= INT64_MIN + padding);
                                // This should never overflow because the 4-product does not
                                auto ii = (ssize_t)i * stride - padding + di; 
                                auto jj = (ssize_t)j * stride - padding + dj; 

                                if(ii < 0 || jj < 0 || 
                                    ii >= H_in || jj >= W_in) continue; 
                                
                                auto input_idx = (((ssize_t)b * C_in + ic) * H_in + ii) * W_in + jj; 
                                auto weight_idx = (((ssize_t)oc * C_in + ic) * K + di) * K + dj; 

                                // printf("Hello 2: %ld, %ld, %ld, %ld\n", input_idx, weight_idx, (ssize_t)B * C_in * H_in * W_in, (ssize_t)C_out * C_in * K * K); 
                                mydeviceassert(0 <= input_idx && input_idx < (ssize_t)B * C_in * H_in * W_in); 
                                mydeviceassert(0 <= weight_idx && weight_idx < (ssize_t)C_out * C_in * K * K); 
                                if(dead_input != nullptr && dead_input[input_idx]) continue;
                                // printf("hello1\n");
                                // printf("%ld, %ld\n", input_idx, weight_idx); 
                                auto val = input[input_idx] + weight[weight_idx]; 
                                // printf("hello2\n"); 
                                if(val > res) {
                                    res = val; 
                                    argin = input_idx; 
                                    argw = weight_idx; 
                                    is_dead = false;
                                }
                            }
                        }
                    }
                    // printf("Hello 3: %ld, %ld\n", oc, C_out);
                    mydeviceassert(0 <= oc && oc < C_out); 
                    if(bias != nullptr && bias[oc] > res) {
                        res = bias[oc];
                        argin = -1;   
                        argw = -1;
                        is_dead = false;
                    }

                    // printf("Hello 4: %ld, %ld\n", out_idx, (((ssize_t)B * C_out * H_out * W_out))); 
                    mydeviceassert(0 <= out_idx && out_idx < (((ssize_t)B * C_out * H_out * W_out))); 
                    output[out_idx] = res; 
                    argmax_input[out_idx] = argin; 
                    argmax_weight[out_idx] = argw; 
                    if(dead_input != nullptr) dead_output[out_idx] = is_dead; 
                }
            }
        }
        __syncthreads(); 
    }

__global__ void maxplus_conv2d_forward_gpu_kernel(
    const c10_double* input, const c10_double* weight, const c10_double* bias, const bool* dead_input,
    int* error_flag, 
    int64_t K, int64_t stride, int64_t padding, int64_t B, int64_t C_in, int64_t C_out,
    int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out, 
    c10_double* output, ssize_t* argmax_input, ssize_t* argmax_weight, bool* dead_output) {

        if(*error_flag) return; 

        __syncthreads();
        for(ssize_t i = threadIdx.y + blockDim.y * blockIdx.y; i < (ssize_t)H_out; i += gridDim.y * blockDim.y) {
            for(ssize_t j = threadIdx.z + blockDim.z * blockIdx.z; j < (ssize_t)W_out; j += gridDim.z * blockDim.z) {
                for(ssize_t out_idx_bc = threadIdx.x + blockDim.x * blockIdx.x; out_idx_bc < (ssize_t)B * C_out; out_idx_bc += gridDim.x * blockDim.x) {
                    auto out_idx = (out_idx_bc * H_out + i) * W_out + j; 
                    int64_t oc = out_idx_bc % C_out; 
                    int64_t b = out_idx_bc / C_out; 
                    mydeviceassert(b >= 0 && oc >= 0 && i >= 0 && j >= 0 &&
                            b < B && oc < C_out && i < H_out && j < W_out); 
                    // Same correctness guarrantee as the serial one, from this point onward everything should
                    // work as is. 
                    c10_double res = -(c10_double)INFINITY; 
                    ssize_t argin = -2;  
                    ssize_t argw = -2; 
                    bool is_dead = true; 
                    for(int64_t ic = 0; ic < C_in; ic++) {
                        for(int64_t di = 0; di < K; di++) {
                            for(int64_t dj = 0; dj < K; dj++) {
                                mydeviceassert((ssize_t)i * stride >= INT64_MIN + padding);
                                mydeviceassert((ssize_t)j * stride >= INT64_MIN + padding);
                                // This should never overflow because the 4-product does not
                                auto ii = (ssize_t)i * stride - padding + di; 
                                auto jj = (ssize_t)j * stride - padding + dj; 

                                if(ii < 0 || jj < 0 || 
                                    ii >= H_in || jj >= W_in) continue; 
                                
                                auto input_idx = (((ssize_t)b * C_in + ic) * H_in + ii) * W_in + jj; 
                                auto weight_idx = (((ssize_t)oc * C_in + ic) * K + di) * K + dj; 

                                mydeviceassert(0 <= input_idx && input_idx < (ssize_t)B * C_in * H_in * W_in); 
                                mydeviceassert(0 <= weight_idx && weight_idx < (ssize_t)C_out * C_in * K * K); 
                                if(dead_input != nullptr && dead_input[input_idx]) continue;
                                // printf("hello1\n");
                                // printf("%ld, %ld\n", input_idx, weight_idx); 
                                auto val = input[input_idx] + weight[weight_idx]; 
                                // printf("hello2\n"); 
                                if(val > res) {
                                    res = val; 
                                    argin = input_idx; 
                                    argw = weight_idx; 
                                    is_dead = false;
                                }
                            }
                        }
                    }
                    mydeviceassert(0 <= oc && oc < C_out); 
                    if(bias != nullptr && bias[oc] > res) {
                        res = bias[oc];
                        argin = -1;   
                        argw = -1;
                        is_dead = false;
                    }

                    mydeviceassert(0 <= out_idx && out_idx < (((ssize_t)B * C_out * H_out * W_out))); 
                    output[out_idx] = res; 
                    argmax_input[out_idx] = argin; 
                    argmax_weight[out_idx] = argw; 
                    if(dead_input != nullptr) dead_output[out_idx] = is_dead; 
                }
            }
        }
        __syncthreads(); 
    }

__global__ void maxplus_conv2d_backward_gpu_kernel(
    const c10_float* grad_output, const ssize_t* argmax_input, const ssize_t* argmax_weight, const bool* dead_output, 
    int* error_flag, 
    int64_t K, int64_t B, int64_t C_in, int64_t C_out,
    int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out, 
    c10_float* grad_input, c10_float* grad_weight, c10_float* grad_bias = nullptr) {

        if(*error_flag) return; 

        __syncthreads();
        for(ssize_t i = threadIdx.y + blockDim.y * blockIdx.y; i < (ssize_t)H_out; i += gridDim.y * blockDim.y) {
            for(ssize_t j = threadIdx.z + blockDim.z * blockIdx.z; j < (ssize_t)W_out; j += gridDim.z * blockDim.z) {
                for(ssize_t out_idx_bc = threadIdx.x + blockDim.x * blockIdx.x; out_idx_bc < (ssize_t)B * C_out; out_idx_bc += gridDim.x * blockDim.x) {
                    auto out_idx = (out_idx_bc * H_out + i) * W_out + j; 
                    int64_t oc = out_idx_bc % C_out; 
                    int64_t b = out_idx_bc / C_out; 
                    // printf("Hello 1: %ld, %ld, %ld, %ld\n", b, oc, i, j);
                    mydeviceassert(b >= 0 && oc >= 0 && i >= 0 && j >= 0 &&
                            b < B && oc < C_out && i < H_out && j < W_out); 
                    
                    // printf("Hello 2: %ld, %ld\n", out_idx, (((ssize_t)B * C_out * H_out * W_out)));
                    mydeviceassert(0 <= out_idx && out_idx < (((ssize_t)B * C_out * H_out * W_out)));
                    auto go = grad_output[out_idx]; 

                    if(dead_output != nullptr && dead_output[out_idx]) continue;
                    auto argin = argmax_input[out_idx]; 
                    auto argw = argmax_weight[out_idx]; 
                    
                    // printf("Hello 3: %ld, %ld\n", argin, argw);
                    mydeviceassert(argin > -2 && argw > -2); 

                    if(argin != -1 && argw != -1) {
                        // printf("Hello 4: %ld, %ld, %ld, %ld\n", argin, argw, (ssize_t)B * C_in * H_in * W_in, (ssize_t)C_out * C_in * K * K);
                        mydeviceassert(argin < (ssize_t)B * C_in * H_in * W_in && argw < (ssize_t)C_out * C_in * K * K);
                        atomicAdd(&grad_input[argin], go);  
                        atomicAdd(&grad_weight[argw], go); 
                    } else if(argin == -1 && argw == -1 && grad_bias != nullptr) {
                        // printf("Hello 5: %ld, %ld\n", oc, C_out);
                        mydeviceassert(0 <= oc && oc < C_out); 
                        atomicAdd(&grad_bias[oc], go); 
                    } else {
                        // printf("Hello 6\n");
                        mydeviceassert(false); 
                    }
                }
            }
        }
        __syncthreads(); 
    }

__global__ void maxplus_conv2d_backward_gpu_kernel(
    const c10_double* grad_output, const ssize_t* argmax_input, const ssize_t* argmax_weight, const bool* dead_output, 
    int* error_flag, 
    int64_t K, int64_t B, int64_t C_in, int64_t C_out,
    int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out, 
    c10_double* grad_input, c10_double* grad_weight, c10_double* grad_bias = nullptr) {

        if(*error_flag) return; 

        __syncthreads();
        for(ssize_t i = threadIdx.y + blockDim.y * blockIdx.y; i < (ssize_t)H_out; i += gridDim.y * blockDim.y) {
            for(ssize_t j = threadIdx.z + blockDim.z * blockIdx.z; j < (ssize_t)W_out; j += gridDim.z * blockDim.z) {
                for(ssize_t out_idx_bc = threadIdx.x + blockDim.x * blockIdx.x; out_idx_bc < (ssize_t)B * C_out; out_idx_bc += gridDim.x * blockDim.x) {
                    auto out_idx = (out_idx_bc * H_out + i) * W_out + j; 
                    int64_t oc = out_idx_bc % C_out; 
                    int64_t b = out_idx_bc / C_out; 
                    mydeviceassert(b >= 0 && oc >= 0 && i >= 0 && j >= 0 &&
                            b < B && oc < C_out && i < H_out && j < W_out); 

                    mydeviceassert(0 <= out_idx && out_idx < (((ssize_t)B * C_out * H_out * W_out)));
                    auto go = grad_output[out_idx]; 

                    if(dead_output != nullptr && dead_output[out_idx]) continue;
                    auto argin = argmax_input[out_idx]; 
                    auto argw = argmax_weight[out_idx]; 
                    
                    mydeviceassert(argin > -2 && argw > -2); 

                    if(argin != -1 && argw != -1) {
                        mydeviceassert(argin < (ssize_t)B * C_in * H_in * W_in && argw < (ssize_t)C_out * C_in * K * K);
                        atomicAdd(&grad_input[argin], go);  
                        atomicAdd(&grad_weight[argw], go); 
                    } else if(argin == -1 && argw == -1 && grad_bias != nullptr) {
                        mydeviceassert(0 <= oc && oc < C_out); 
                        atomicAdd(&grad_bias[oc], go); 
                    } else mydeviceassert(false); 
                }
            }
        }
        __syncthreads(); 
    }

// Constraint: all tensor work on a common floating type. Similar to pytorch, mismatch raises error.
// Constraint: Torch internally uses signed 64-bit ints for memory indexing. We make the same assumption regarding
//              pointers, and drop size_t in favor of int64_t. 

// Checklist:
/*
1) chech proper dims  X
2) check consistency in dims  X
3) handle null tensors  X
4) check non-negativity and non-triviality  X
5) make as portable to library changes as possible  X
6) check overflow in pointers, i.e., tensors fit in int64_t  X
7) make tensors contiguous if not (no const tensor& unfortunately. avoid copying for speed)  X
8) check proper devices  X
*/

// The function is observer-state unmutable. If the given tensors are contiguous, then it is putrely unmutable with 
// respect to the input. Interal changes possible on input tensors to avoid copying.
std::vector<torch::Tensor> maxplus_conv2d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor dead_input, 
    int64_t stride=1, int64_t padding=0) {

        // Input and weight must not be None
        TORCH_CHECK(input.defined() && weight.defined(), "Input and weight tensors need to be defined. "); 

        auto working_device = input.options().device(); 

        // Check same device
        TORCH_CHECK(weight.options().device() == working_device, "All tensors need to be in the same device: weight.device != input.device. "); 

        // Input = [B, C_out, H_out, W_out], Weight = [C_out, C_in, K, K]. Both 4-dim tensors
        TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "Input and weight tensors need to have 4 dimensions. "); 

        auto B = input.size(0), C_in = input.size(1), H_in = input.size(2), W_in = input.size(3);
        auto C_out = weight.size(0), K = weight.size(2); 

        // Assert consistency in dims
        TORCH_CHECK(C_in == weight.size(1), "Inconsistent dimensions between input and weight. ");

        // Check whether bias and dead_input are used
        bool use_bias = bias.defined(); 
        bool use_dead = dead_input.defined(); 

        // If yes, perform the same checks for dims, consistency, and device
        if(use_bias) {
            TORCH_CHECK(bias.dim() == 1, "Bias needs to have 1 dimension. ");
            TORCH_CHECK(bias.size(0) == C_out, "Inconsistent bias dimension != C_out. "); 
            TORCH_CHECK(bias.options().device() == working_device, "All tensors need to be in the same device: bias.device != input.device. "); 
        }
        if(use_dead) {
            TORCH_CHECK(dead_input.dim() == 4, "dead_input needs to have 4 dimension. "); 
            TORCH_CHECK(dead_input.size(0) == input.size(0) && dead_input.size(1) == input.size(1) &&
                    dead_input.size(2) == input.size(2) && dead_input.size(3) == input.size(3), "Inconsistent input and dead_input dimensions. "); 
            TORCH_CHECK(dead_input.options().device() == working_device, "All tensors need to be in the same device: dead_input.device != input.device. ");
        }
        
        // Ensure tensors are well-defined and non-trivial. 
        TORCH_CHECK(B >= 1 && C_in >= 1 && H_in >= 1 && W_in >= 1 && K >= 1, "A dimension is either not well-defined or 0. "); 

        // Ensure memory references using pytorch's int64_t never overflow for input and weight. 
        TORCH_CHECK(B < INT64_MAX / C_in && B * C_in < INT64_MAX / H_in && B * C_in * H_in < INT64_MAX / W_in, "int64_t overflow. "); 
        TORCH_CHECK(C_out < INT64_MAX / C_in && C_out * C_in < INT64_MAX / K && C_out * C_in * K < INT64_MAX / K, "int64_t overflow. "); 
        
        // Get the working type of the tensors. It should be a floating type
        auto working_type = input.options().dtype();  
        TORCH_CHECK(c10::isFloatingType(c10::typeMetaToScalarType(working_type)), "The working type should be a floating type. "); 

        // All tensors must use the same type
        TORCH_CHECK(weight.options().dtype() == working_type, "All tensors must use the same type. weight.type != input.typ. ");  

        if(use_bias) TORCH_CHECK(bias.options().dtype() == working_type, "All tensors must use the same type. bias.type != input.type. "); 
        if(use_dead) TORCH_CHECK(dead_input.options().dtype() == torch::kBool, "Dead.type must be of type bool. "); 

        // Ensure the input tensors are contiguous in memory
        if(!input.is_contiguous()) {
            input = input.contiguous(); 
        }
        if(!weight.is_contiguous()) {
            weight = weight.contiguous(); 
        }
        if(use_bias && !bias.is_contiguous()) {
            bias = bias.contiguous(); 
        }
        if(use_dead && !dead_input.is_contiguous()) {
            dead_input = dead_input.contiguous(); 
        }

        // Ensure no overflow for output channel resolution dims. 
        TORCH_CHECK(H_in >= 1*(K-1) + INT64_MIN && H_in - 1*(K-1) <= INT64_MAX - 2*padding, "int64_t overflow. ");
        if((H_in + 2*padding - 1*(K-1)) % stride != 0) TORCH_CHECK((H_in - 1*(K-1) + 2*padding) / stride < INT64_MAX, "int64_t overflow. ");
        auto H_out = (H_in - 1*(K-1) + 2*padding) / stride + ((H_in + 2*padding - 1*(K-1)) % stride != 0); 

        TORCH_CHECK(W_in >= 1*(K-1) + INT64_MIN && W_in - 1*(K-1) <= INT64_MAX - 2*padding, "int64_t overflow. ");
        if((W_in + 2*padding - 1*(K-1)) % stride != 0) TORCH_CHECK((W_in - 1*(K-1) + 2*padding) / stride < INT64_MAX, "int64_t overflow. ");
        auto W_out = (W_in - 1*(K-1) + 2*padding) / stride + ((W_in + 2*padding - 1*(K-1)) % stride != 0); 

        TORCH_CHECK(H_out >= 1 && W_out >= 1, "Output image reduces to 0 pixels. "); 

        // Ensure output tensor can be indexed with int64_t
        TORCH_CHECK(B < INT64_MAX / C_out && B * C_out < INT64_MAX / H_out && B * C_out * H_out < INT64_MAX / W_out, "int64_t overflow. "); 

        // Define outputs. Careful to keep the same options as input
        auto output = torch::full({B, C_out, H_out, W_out}, -INFINITY, input.options().dtype(working_type));  
        auto argmax_input = torch::zeros_like(output, torch::kInt64); 
        auto argmax_weight = torch::zeros_like(output, torch::kInt64);
        // Empty initilized tensors, or empty initializer lists {}, map to python None.  
        auto dead_output = (use_dead ? torch::zeros_like(output, torch::kBool) : torch::Tensor()); 

        // Error flag raised by any thread of the kernels to indicate an error
        int* error_flag; 

        // Different cpu fallback "kernels" depending on the working type of the tensors.
        if(working_device.is_cpu()) {
            error_flag = (int*)malloc(sizeof(int));  // this is unnecessary, but I add it to match the calls
            *error_flag = 0; 
            switch(c10::typeMetaToScalarType(working_type)) {
                case torch::kFloat:
                    maxplus_conv2d_forward_cpu_fallback(
                        input.data_ptr<c10_float>(), weight.data_ptr<c10_float>(),  
                        (use_bias ? bias.data_ptr<c10_float>() : nullptr), 
                        (use_dead ? dead_input.data_ptr<bool>() : nullptr),
                        error_flag, 
                        K, stride, padding, B, C_in, C_out, H_in, W_in, H_out, W_out, 
                        output.data_ptr<c10_float>(), argmax_input.data_ptr<ssize_t>(), argmax_weight.data_ptr<ssize_t>(), 
                        (use_dead ? dead_output.data_ptr<bool>() : nullptr)
                    ); 
                    if(*error_flag) {
                        free(error_flag); 
                        throw(-1);
                    }
                    free(error_flag); 
                    break;
                case torch::kDouble:
                    maxplus_conv2d_forward_cpu_fallback(
                        input.data_ptr<c10_double>(), weight.data_ptr<c10_double>(),  
                        (use_bias ? bias.data_ptr<c10_double>() : nullptr), 
                        (use_dead ? dead_input.data_ptr<bool>() : nullptr),
                        error_flag, 
                        K, stride, padding, B, C_in, C_out, H_in, W_in, H_out, W_out, 
                        output.data_ptr<c10_double>(), argmax_input.data_ptr<ssize_t>(), argmax_weight.data_ptr<ssize_t>(), 
                        (use_dead ? dead_output.data_ptr<bool>() : nullptr)
                    ); 
                    if(*error_flag) {
                        free(error_flag); 
                        throw(-1); 
                    }
                    free(error_flag); 
                    break; 
                free(error_flag); 
                TORCH_CHECK(false, "Types other than float and double not supported. ");
            }
        } 
        else if(working_device.is_cuda()) {
            cudaMalloc(&error_flag, sizeof(int)); 
            int error_flag_host = 0; cudaMemcpy(error_flag, &error_flag_host, sizeof(int), cudaMemcpyHostToDevice); 
            dim3 threads(1, THREADS_PER_BLOCK, THREADS_PER_BLOCK); 
            const auto f = (STRIDE * THREADS_PER_BLOCK);
            dim3 blocks((B * C_out + f - 1) / f, (H_out + f - 1) / f, (W_out + f - 1) / f); 
            switch(c10::typeMetaToScalarType(working_type)) {
                case torch::kFloat:
                    // printf("Kernel entry");
                    maxplus_conv2d_forward_gpu_kernel<<<blocks, threads>>>(
                        input.data_ptr<c10_float>(), weight.data_ptr<c10_float>(),  
                        (use_bias ? bias.data_ptr<c10_float>() : nullptr), 
                        (use_dead ? dead_input.data_ptr<bool>() : nullptr),
                        error_flag, 
                        K, stride, padding, B, C_in, C_out, H_in, W_in, H_out, W_out, 
                        output.data_ptr<c10_float>(), argmax_input.data_ptr<ssize_t>(), argmax_weight.data_ptr<ssize_t>(), 
                        (use_dead ? dead_output.data_ptr<bool>() : nullptr)
                    ); 
                    cudaDeviceSynchronize(); 
                    cudaMemcpy(&error_flag_host, error_flag, sizeof(int), cudaMemcpyDeviceToHost); 
                    if(error_flag_host) {
                        cudaFree(error_flag); 
                        throw(-1); 
                    }
                    cudaFree(error_flag); 
                    break;
                case torch::kDouble:
                    maxplus_conv2d_forward_gpu_kernel<<<blocks, threads>>>(
                        input.data_ptr<c10_double>(), weight.data_ptr<c10_double>(),  
                        (use_bias ? bias.data_ptr<c10_double>() : nullptr), 
                        (use_dead ? dead_input.data_ptr<bool>() : nullptr),
                        error_flag, 
                        K, stride, padding, B, C_in, C_out, H_in, W_in, H_out, W_out, 
                        output.data_ptr<c10_double>(), argmax_input.data_ptr<ssize_t>(), argmax_weight.data_ptr<ssize_t>(), 
                        (use_dead ? dead_output.data_ptr<bool>() : nullptr)
                    ); 
                    cudaDeviceSynchronize(); 
                    cudaMemcpy(&error_flag_host, error_flag, sizeof(int), cudaMemcpyDeviceToHost); 
                    if(error_flag_host) {
                        cudaFree(error_flag); 
                        throw(-1); 
                    }
                    cudaFree(error_flag); 
                    break; 
                cudaFree(error_flag); 
                TORCH_CHECK(false, "Types other than float and double not supported. ");
            }
        }
        else {
            TORCH_CHECK(false, "Devices other than CPU and CUDA not supported. ");
        }
        // the move constructors should be called without me specifying it. destruction will happen internally in python
        // once the moved object falls out of scope. 
        return {output, argmax_input, argmax_weight, dead_output}; 
    }

// Checklist:
/*
1) chech proper dims  X
2) check consistency in dims  X
3) handle null tensors  X
4) check non-negativity and non-triviality  X
5) make as portable to library changes as possible  X
6) check overflow in pointers, i.e., tensors fit in int64_t  X
7) make tensors contiguous if not (no const tensor& unfortunately. avoid copying for speed)  X
8) check proper devices  X
*/

std::vector<torch::Tensor> maxplus_conv2d_backward(
    torch::Tensor grad_output, torch::Tensor argmax_input, torch::Tensor argmax_weight, torch::Tensor dead_output, 
    int64_t C_in, int64_t H_in, int64_t W_in, int64_t K, int64_t stride=1, int64_t padding=0) {

        // grad_output and argmax's must not be None
        TORCH_CHECK(grad_output.defined() && argmax_input.defined() && argmax_weight.defined(), "Input and weight tensors need to be defined. "); 

        auto working_device = grad_output.options().device(); 

        // Check same device
        TORCH_CHECK(argmax_input.options().device() == working_device, "All tensors need to be in the same device: argmax_input.device != grad_output.device. "); 
        TORCH_CHECK(argmax_weight.options().device() == working_device, "All tensors need to be in the same device: argmax_weight.device != grad_output.device. "); 

        // input tensors must have dims [B, C_out, H_out, W_out], same as output of forward
        TORCH_CHECK(grad_output.dim() == 4 && argmax_input.dim() == 4 && argmax_weight.dim() == 4, "Input and weight tensors need to have 4 dimensions. "); 

        int64_t B = grad_output.size(0), C_out = grad_output.size(1), H_out = grad_output.size(2), W_out = grad_output.size(3); 

        TORCH_CHECK(argmax_input.size(0) == B && argmax_input.size(1) == C_out 
                    && argmax_input.size(2) == H_out && argmax_input.size(3) == W_out, "grad_output and argmax_input dims must be consistent. ")

        TORCH_CHECK(argmax_weight.size(0) == B && argmax_weight.size(1) == C_out 
                    && argmax_weight.size(2) == H_out && argmax_weight.size(3) == W_out, "grad_output and argmax_weight dims must be consistent. ")


        // Check the same for dead_output, if used. 
        bool use_dead = dead_output.defined(); 
        if(use_dead) {
            TORCH_CHECK(dead_output.dim() == 4, "dead_output needs to have 4 dimension. "); 
            TORCH_CHECK(dead_output.size(0) == grad_output.size(0) && dead_output.size(1) == grad_output.size(1) &&
                    dead_output.size(2) == grad_output.size(2) && dead_output.size(3) == grad_output.size(3), 
                "dead_output needs to have be consistent with grad_output dims. "); 
            // TORCH_CHECK(dead_output.options().device().is_cpu());
            TORCH_CHECK(dead_output.options().device() == working_device, "dead_output needs to be in the device. "); 
        }

        // Ensure non-negativity and non-triviality of output dims
        TORCH_CHECK(B >= 1 && C_out >= 1 && H_out >= 1 && W_out >= 1, "A dimension is either not well-defined or 0. "); 
        // Ensure the same for the dims of the arguments
        TORCH_CHECK(C_in >= 1 && H_in >= 1 && W_in >= 1 && K >= 1, "A dimension is either not well-defined or 0. "); 

        // Ensure dims fit in int64_t. Again, with some slack
        TORCH_CHECK(B < INT64_MAX / C_out && B * C_out < INT64_MAX / H_out && B * C_out * H_out < INT64_MAX / W_out, "int64_t overflow. "); 
        TORCH_CHECK(B < INT64_MAX / C_in && B * C_in < INT64_MAX / H_in && B * C_in * H_in < INT64_MAX / W_in, "int64_t overflow. "); 
        TORCH_CHECK(C_out < INT64_MAX / C_in && C_out * C_in < INT64_MAX / K && C_out * C_in * K < INT64_MAX / K, "int64_t overflow. "); 

        // Ensure consistency with input and weight dims provided in the arguments. 
        TORCH_CHECK(H_in >= 1*(K-1) + INT64_MIN && H_in - 1*(K-1) <= INT64_MAX - 2*padding, "int64_t overflow. ");
        if((H_in + 2*padding - 1*(K-1)) % stride != 0) TORCH_CHECK((H_in - 1*(K-1) + 2*padding) / stride < INT64_MAX, "int64_t overflow. ");
        auto H_out_comp = (H_in - 1*(K-1) + 2*padding) / stride + ((H_in + 2*padding - 1*(K-1)) % stride != 0); 

        TORCH_CHECK(W_in >= 1*(K-1) + INT64_MIN && W_in - 1*(K-1) <= INT64_MAX - 2*padding, "int64_t overflow. ");
        if((W_in + 2*padding - 1*(K-1)) % stride != 0) TORCH_CHECK((W_in - 1*(K-1) + 2*padding) / stride < INT64_MAX, "int64_t overflow. ");
        auto W_out_comp = (W_in - 1*(K-1) + 2*padding) / stride + ((W_in + 2*padding - 1*(K-1)) % stride != 0); 

        TORCH_CHECK(H_out_comp == H_out && W_out_comp == W_out, "Given input and weight dims not consistent with grad_output dims. "); 

        // Get working type of grad_output. Must be a floating type. 
        auto working_type = grad_output.options().dtype();  
        TORCH_CHECK(c10::isFloatingType(c10::typeMetaToScalarType(working_type)), "The working type should be a floating type. "); 
        TORCH_CHECK(argmax_input.options().dtype() == torch::kInt64, "argmax_input should be Int64. ");  
        TORCH_CHECK(argmax_weight.options().dtype() == torch::kInt64, "argmax_weight should be Int64. "); 

        if(use_dead) TORCH_CHECK(dead_output.options().dtype() == torch::kBool, "dead_output should be Bool. "); 

        // Ensure input tensors are contiguous so that sequencial memory accesses in the kernels makes sense. 
        if(!grad_output.is_contiguous()) {
            grad_output = grad_output.contiguous(); 
        }
        if(!argmax_input.is_contiguous()) {
            argmax_input = argmax_input.contiguous(); 
        }
        if(!argmax_weight.is_contiguous()) {
            argmax_weight = argmax_weight.contiguous(); 
        }
        if(use_dead && !dead_output.is_contiguous()) {
            dead_output = dead_output.contiguous(); 
        }

        // Define output tensors (back gradients). 
        auto grad_input = torch::zeros({B, C_in, H_in, W_in}, grad_output.options());  
        auto grad_weight = torch::zeros({C_out, C_in, K, K}, grad_output.options()); 
        auto grad_bias = torch::zeros({C_out}, grad_output.options());

        // Error flag raised by any thread of the kernels to indicate an error
        int* error_flag; 

        // Different cpu fallback "kernels" depending on the working type of the tensors.
        if(working_device.is_cpu()) {
            error_flag = (int*)malloc(sizeof(int)); 
            *error_flag = 0; 
            switch(c10::typeMetaToScalarType(working_type)) {
                case torch::kFloat:
                    maxplus_conv2d_backward_cpu_fallback(
                        grad_output.data_ptr<c10_float>(), argmax_input.data_ptr<ssize_t>(), argmax_weight.data_ptr<ssize_t>(),
                        (use_dead ? dead_output.data_ptr<bool>() : nullptr), 
                        error_flag, 
                        K, B, C_in, C_out, H_in, W_in, H_out, W_out, 
                        grad_input.data_ptr<c10_float>(), grad_weight.data_ptr<c10_float>(), grad_bias.data_ptr<c10_float>() 
                    ); 
                    if(*error_flag) {
                        free(error_flag); 
                        throw(-1); 
                    }
                    free(error_flag); 
                    break;
                case torch::kDouble:
                    maxplus_conv2d_backward_cpu_fallback(
                        grad_output.data_ptr<c10_double>(), argmax_input.data_ptr<ssize_t>(), argmax_weight.data_ptr<ssize_t>(),
                        (use_dead ? dead_output.data_ptr<bool>() : nullptr), 
                        error_flag, 
                        K, B, C_in, C_out, H_in, W_in, H_out, W_out, 
                        grad_input.data_ptr<c10_double>(), grad_weight.data_ptr<c10_double>(), grad_bias.data_ptr<c10_double>() 
                    ); 
                    if(*error_flag) {
                        free(error_flag); 
                        throw(-1); 
                    }
                    free(error_flag); 
                    break;
                free(error_flag); 
                TORCH_CHECK(false, "Types other than float and double not supported. ");
            }
        }
        else if(working_device.is_cuda()) {
            cudaMalloc(&error_flag, sizeof(int));  
            int error_flag_host = 0; cudaMemcpy(error_flag, &error_flag_host, sizeof(int), cudaMemcpyHostToDevice); 
            dim3 threads(THREADS_PER_BLOCK); 
            const auto f = (STRIDE * THREADS_PER_BLOCK);
            dim3 blocks((B * C_out + f - 1) / f, (H_out + f - 1) / f, (W_out + f - 1) / f); 
            switch(c10::typeMetaToScalarType(working_type)) {
                case torch::kFloat:
                    // printf("Kernel entry");
                    maxplus_conv2d_backward_gpu_kernel<<<blocks, threads>>>(
                        grad_output.data_ptr<c10_float>(), argmax_input.data_ptr<ssize_t>(), argmax_weight.data_ptr<ssize_t>(),
                        (use_dead ? dead_output.data_ptr<bool>() : nullptr), 
                        error_flag, 
                        K, B, C_in, C_out, H_in, W_in, H_out, W_out, 
                        grad_input.data_ptr<c10_float>(), grad_weight.data_ptr<c10_float>(), grad_bias.data_ptr<c10_float>() 
                    ); 
                    cudaDeviceSynchronize(); 
                    cudaMemcpy(&error_flag_host, error_flag, sizeof(int), cudaMemcpyDeviceToHost); 
                    if(error_flag_host) {
                        cudaFree(error_flag); 
                        throw(-1); 
                    }
                    cudaFree(error_flag); 
                    break;
                case torch::kDouble:
                    maxplus_conv2d_backward_gpu_kernel<<<blocks, threads>>>(
                        grad_output.data_ptr<c10_double>(), argmax_input.data_ptr<ssize_t>(), argmax_weight.data_ptr<ssize_t>(),
                        (use_dead ? dead_output.data_ptr<bool>() : nullptr), 
                        error_flag, 
                        K, B, C_in, C_out, H_in, W_in, H_out, W_out, 
                        grad_input.data_ptr<c10_double>(), grad_weight.data_ptr<c10_double>(), grad_bias.data_ptr<c10_double>() 
                    ); 
                    cudaDeviceSynchronize(); 
                    cudaMemcpy(&error_flag_host, error_flag, sizeof(int), cudaMemcpyDeviceToHost); 
                    if(error_flag_host) {
                        cudaFree(error_flag); 
                        throw(-1); 
                    }
                    cudaFree(error_flag); 
                    break;
                cudaFree(error_flag); 
                TORCH_CHECK(false, "Types other than float and double not supported. ");
            }
        } 
        else {
            TORCH_CHECK(false, "Devices other than CPU and GPU not supported. ");
        }
        // the move constructors should be called without me specifying it. destruction will happen internally in python
        // once the moved object falls out of scope. 
        return {grad_input, grad_weight, grad_bias}; 
    } 

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxplus_conv2d_forward", &maxplus_conv2d_forward, "Max-Plus Conv2D Forward (CUDA)");
    m.def("maxplus_conv2d_backward", &maxplus_conv2d_backward, "Max-Plus Conv2D Backward (CUDA)");
}