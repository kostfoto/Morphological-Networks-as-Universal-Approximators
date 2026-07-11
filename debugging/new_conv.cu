#include <bits/stdc++.h>
// #include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
// It is GPU dependent, not sqrt. 
#define STEP 100000000000

// float INF = std::numeric_limits<float>::infinity(); 

float INF = INFINITY

// Constraint 1: C_in, C_out, K, stride, padding, resol_j = H_j * W_j should all fit in int32_t. 
// Constraint 2: batch size B should fit in int32_t
// Constraint 3: int32_t should be 4-bytes. 
// Constraint 4: all array indices should fit in ssize_t 

// Bad-practice 1: I won't use shorts to record dead/weight/bias. I want to save memory. I will
//                  use bool for dead/alive, and -2/-1/>=0 in the argmax's for dead/bias/weight

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts); 
    return ts.tv_sec + ts.tv_nsec * 1e-9; 
}

struct Timer {

    double start, end; 
    double duration; 
    int id; 

    Timer(int id) {
        start = get_time();
        this->id = id; 
    }

    ~Timer() {
        end = get_time();
        duration = end - start; 

        double ms = duration * 1000.0f; 

        printf("Timer %d took %.3lf (ms).\n", id, ms); 
    }
};

// fixed dilation = 1 and groups = 1. Other options not-supported. 
void morph_conv_2d_forward_cpu_fallback(
    const float* input, const float* weight, const float* bias, const bool* dead_input,
    int32_t K, int32_t stride, int32_t padding, int32_t B, int32_t C_in, int32_t C_out,
    int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out, 
    float* output, ssize_t* argmax_input, ssize_t* argmax_weight, bool* dead_output) {

        for(int32_t b = 0; b < B; b++) {
            for(int32_t oc = 0; oc < C_out; oc++) {
                for(int32_t i = 0; i < H_out; i++) {
                    for(int32_t j = 0; j < W_out; j++) {
                        // This never overflows by assumption
                        ssize_t out_idx = (((ssize_t)b * C_out + oc) * H_out + i) * W_out + j; 
                        float res = -INF; 
                        ssize_t argin = -2;  
                        ssize_t argw = -2; 
                        bool is_dead = true; 
                        for(int32_t ic = 0; ic < C_in; ic++) {
                            for(int32_t di = 0; di < K; di++) {
                                for(int32_t dj = 0; dj < K; dj++) {
                                    // This should never overflow (int32 * int32 -> int64)
                                    auto ii = (ssize_t)i * stride + di - padding; 
                                    auto jj = (ssize_t)j * stride + dj - padding; 

                                    if(ii < 0 || jj < 0 || 
                                        ii >= H_in || jj >= W_in) continue; 
                                    
                                    auto input_idx = (((ssize_t)b * C_in + ic) * H_in + ii) * W_in + jj; 
                                    auto weight_idx = (((ssize_t)oc * C_in + ic) * K + di) * K + dj; 

                                    assert(0 <= input_idx && input_idx < (ssize_t)B * C_in * H_in * W_in); 
                                    assert(0 <= weight_idx && weight_idx < (ssize_t)C_out * C_in * K * K); 
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
                        if(bias != nullptr && bias[oc] > res) {
                            res = bias[oc];
                            argin = -1;   
                            argw = -1;
                            is_dead = false;
                        }

                        output[out_idx] = res; 
                        argmax_input[out_idx] = argin; 
                        argmax_weight[out_idx] = argw; 
                        dead_output[out_idx] = is_dead; 
                    }
                }
            }
        }
    }

void morph_conv_2d_backward_cpu_fallback(
    const float* grad_output, const ssize_t* argmax_input, const ssize_t* argmax_weight, const bool* dead_output, 
    int32_t K, int32_t B, int32_t C_in, int32_t C_out,
    int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out, 
    float* grad_input, float* grad_weight, float* grad_bias = nullptr) {

        for(int32_t oc = 0; oc < C_out; oc++) {
            for(int32_t i = 0; i < H_out; i++) {
                for(int32_t j = 0; j < W_out; j++) {
                    auto out_idx = ((ssize_t)oc * H_out + i) * W_out + j; 
                    auto go = grad_output[out_idx]; 

                    if(dead_output != nullptr && dead_output[out_idx]) continue;
                    auto argin = argmax_input[out_idx]; 
                    auto argw = argmax_weight[out_idx]; 
                    
                    assert(argin > -2 && argw > -2); 

                    if(argin != -1 && argw != -1) {
                        assert(argin < (ssize_t)B * C_in * H_in * W_in && argw < (ssize_t)C_out * C_in * K * K);
                        grad_input[argin] += go; 
                        grad_weight[argw] += go; 
                    } else if(argin == -1 && argw == -1 && grad_bias != nullptr) {
                        grad_bias[oc] += go; 
                    } else assert(false); 
                }
            }
        }
    }

// =========================================================================
// Option 0:  What I was doing already
// Option 1:  Output-pixel level parallelism, no stride
// Threads = 256, Blocks = #out_pixels / 256 = B * C_out * H_out * W_out / 256 < 64 * 256 * 256 * 256 / 256 = 1e9 / 256
// - blocks in the order of 10M. 
// - for loop over in_chns, in_pixels: C_in * K * K < 256 * 13 * 13 ~= 50,000. 
// - 50,000 leads to timeout of threads after 2s (how???). 
// + strong scaling, no time wasted synchronizing. 
// 
// Option 2: Output-pixel level parallelism, stride
// Threads = 256, Blocks = stride / 256 = 1e8 / 256
// - blocks in the order of 1M. 
// balanced stride = sqrt(1e9 * 256 * 50,000) ~= 1e7
// - for loop over out_pixels/stride x in_chns x in_pixels: (B * C_out * H_out * W_out / stride) * C_in * K * K ~= 5,000,000
// - 5,000,000, and this will somehow not lead to timeout. 
// + strong scaling, no time wasted synchronizing. 
// 
// Option 3: Input-pixel level parallelism, stride
// Threads = 256, Blocks = (B * C_out * H_out * W_out * stride) / 256 = (1e9 * stride) / 256
// - blocks in the order of 1B. 
// - for loop over in_chns x in_pixels / stride: C_in * K * K / stride ~= 50,000 / stride
// - weak scaling, synchronization needed for calculation for maximum in a binary tree (a mess). 
// 
// 
// Everything depends on how many loops each thread should perform. 
// 
// Option 0 has GPU under-utilization. The block are too few and dont cover all of SMs. And the below. 
// Option 1 will spend a lot of time scheduling the blocks because its thread performs too few instructions. 
// Option 3 will spend a lot of time synchronizing the threads for calculation of maximum. And also scheduling. 
// Option 2 will probably be the best. 
// 
// The only comparative factor is whether option 3 will make up for lost time by calculating the maximum in a binary tree. 
// I doubt it. 
// 
// <TODO> see what block/SM means (in code)
// ==========================================================================


// Assert positivity in wrapper function.

__global__ void morph_conv_2d_forward_gpu_kernel(
    const float* input, const float* weight, const float* bias, const bool* dead_input,
    int32_t K, int32_t stride, int32_t padding, int32_t B, int32_t C_in, int32_t C_out,
    int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out, 
    float* output, ssize_t* argmax_input, ssize_t* argmax_weight, bool* dead_output) {

        __syncthreads();
        for(ssize_t out_idx = threadIdx.x + gridDim.x * blockIdx.x; out_idx < (ssize_t)B*C_out*H_out*W_out; out_idx += gridDim.x * blockDim.x) {
            int32_t j = out_idx % W_out; 
            int32_t i = (out_idx - j) / W_out % H_out; 
            int32_t oc = ((out_idx - j) / W_out - i) / H_out % C_out; 
            int32_t b = (((out_idx - j) / W_out - i) / H_out - oc) / C_out; 
            // Same correctness guarrantee as the serial one, from this point onward everything should
            // work as is. 
            assert(b >= 0 && oc >= 0 && i >= 0 && j >= 0 &&
                    b < B && oc < C_out && i < H_out && j < W_out); 
            float res = -INF; 
            ssize_t argin = -2;  
            ssize_t argw = -2; 
            bool is_dead = true; 
            for(int32_t ic = 0; ic < C_in; ic++) {
                for(int32_t di = 0; di < K; di++) {
                    for(int32_t dj = 0; dj < K; dj++) {
                        // This should never overflow (int32 * int32 -> int64)
                        auto ii = (ssize_t)i * stride + di - padding; 
                        auto jj = (ssize_t)j * stride + dj - padding; 

                        if(ii < 0 || jj < 0 || 
                            ii >= H_in || jj >= W_in) continue; 
                        
                        auto input_idx = (((ssize_t)b * C_in + ic) * H_in + ii) * W_in + jj; 
                        auto weight_idx = (((ssize_t)oc * C_in + ic) * K + di) * K + dj; 

                        // Correctness of argmax's
                        assert(0 <= input_idx && input_idx < (ssize_t)B * C_in * H_in * W_in); 
                        assert(0 <= weight_idx && weight_idx < (ssize_t)C_out * C_in * K * K); 
                        if(dead_input != nullptr && dead_input[input_idx]) continue;
                        auto val = input[input_idx] + weight[weight_idx]; 
                        // No need for synchronization
                        if(val > res) {
                            res = val; 
                            argin = input_idx; 
                            argw = weight_idx; 
                            is_dead = false;
                        }
                    }
                }
            }
            // Again no need for synch. bias[oc] is only read by multiple threads. 
            if(bias != nullptr && bias[oc] > res) {
                res = bias[oc];
                argin = -1;   
                argw = -1;
                is_dead = false;
            }

            output[out_idx] = res; 
            argmax_input[out_idx] = argin; 
            argmax_weight[out_idx] = argw; 
            dead_output[out_idx] = is_dead; 
        }        
        __syncthreads();            
    }

__global__ void morph_conv_2d_backward_gpu_kernel(
    const float* grad_output, const ssize_t* argmax_input, const ssize_t* argmax_weight, const bool* dead_output, 
    int32_t K, int32_t B, int32_t C_in, int32_t C_out,
    int32_t H_in, int32_t W_in, int32_t H_out, int32_t W_out, 
    float* grad_input, float* grad_weight, float* grad_bias = nullptr) {

        __syncthreads();
        for(ssize_t idx = threadIdx.x + gridDim.x * blockIdx.x; idx < (ssize_t)B*C_out*H_out*W_out; idx += gridDim.x * blockDim.x) {
            int32_t j = idx % W_out; 
            int32_t i = (idx - j) / W_out % H_out; 
            int32_t oc = ((idx - j) / W_out - i) / H_out % C_out; 
            int32_t b = (((idx - j) / W_out - i) / H_out - oc) / C_out; 
            assert(b >= 0 && oc >= 0 && i >= 0 && j >= 0 &&
                    b < B && oc < C_out && i < H_out && j < W_out); 
            
            ssize_t out_idx = idx % ((ssize_t)C_out * H_out * W_out); 
            assert(out_idx >= 0 && out_idx < (ssize_t)C_out * H_out * W_out);
            auto go = grad_output[out_idx]; 

            if(dead_output != nullptr && dead_output[out_idx]) continue;
            auto argin = argmax_input[out_idx]; 
            auto argw = argmax_weight[out_idx]; 
            
            // Should only ever reach this with a dead neuron. 
            assert(argin > -2 && argw > -2); 

            if(argin != -1 && argw != -1) {
                // Case: argmax is given by connection to input
                // Redundant check for argmax's correctness
                assert(argin < (ssize_t)B * C_in * H_in * W_in && argw < (ssize_t)C_out * C_in * K * K);
                // atomicity due to shared memory
                atomicAdd(&grad_input[argin], go); 
                atomicAdd(&grad_weight[argw], go); 
            } else if(argin == -1 && argw == -1 && grad_bias != nullptr) {
                // Case: argmax is given by bias. 
                // Bounded asserted already above. 
                atomicAdd(&grad_bias[oc], go); 
            } else{
                exit(-1);
            } 
        }
        __syncthreads(); 
    }

// assert total dims < size_t / 2 so that it fits in the signed version. 
// assert *stride does not overflow

// These will not go into final code...

void initialize(float* a, int32_t n) {
    for(int32_t i = 0; i < n; i++) {
        a[i] = 1.0*(rand() % 100) / 50; 
    }
}

template <typename T>
void print_arr1(T* a, int32_t n, const char* s = "") {
    if(s[0] != 0) printf("%s", s); 
    printf("["); 
    for(int32_t i = 0; i < n; i++) {
        std::cout << a[i] << ", "; 
    }
    printf("]\n"); 
}

template <typename T>
void print_arr2(T* a, int32_t n, int32_t m, const char* s = "") {
    if(s[0] != 0) printf("%s", s); 
    printf("["); 
    for(int32_t i = 0; i < n; i++) {
        print_arr1<T>(a + i * m, m); 
    }
    printf("]\n"); 
}

template <typename T>
void print_arr3(T* a, int32_t n, int32_t m, int32_t k, const char* s = "") {
    if(s[0] != 0) printf("%s", s); 
    printf("["); 
    for(int32_t i = 0; i < n; i++) {
        print_arr2<T>(a + i * m*k, m, k); 
    }
    printf("]\n"); 
}

template <typename T>
void print_arr4(T* a, int32_t b, int32_t n, int32_t m, int32_t k, const char* s = "") {
    if(s[0] != 0) printf("%s", s); 
    printf("["); 
    for(int32_t i = 0; i < b; i++) {
        print_arr3<T>(a + i * n*m*k, n, m, k); 
    }
    printf("]\n"); 
}

template <typename T>
bool arrequal(const T* a, const T* b, size_t n) {
    for(size_t i = 0; i < n; i++) {
        if(abs(a[i] - b[i]) > 1e-5) return false; 
    }
    return true; 
}

// Small matrices
bool test1() {
    printf("===============  Test 1  ================"); 
    int32_t B = 1, C_in = 1, C_out = 1, H_in = 3, W_in = 3, H_out = 1, W_out = 1, K = 3, stride = 1, padding = 0;
    
    int32_t size_input = B * C_in * H_in * W_in; 
    int32_t size_weight = C_out * C_in * K * K;
    int32_t size_output = B * C_out * H_out * W_out;

    float *input = (float*)malloc(size_input * sizeof(float));
    float *weight = (float*)malloc(size_weight * sizeof(float));
    float *bias = nullptr; 
    bool *dead_input = nullptr;  
    float *output = (float*)malloc(size_output * sizeof(float)); 
    ssize_t *argmax_input = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    ssize_t *argmax_weight = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    bool *dead_output = (bool*)malloc(size_output * sizeof(bool));

    float *grad_output = (float*)malloc(size_output * sizeof(float)); 
    float *grad_input = (float*)malloc(size_input * sizeof(float));
    float *grad_weight = (float*)malloc(size_weight * sizeof(float));
    float *grad_bias = nullptr; 

    float *input_gpu, *weight_gpu, *bias_gpu = nullptr; 
    bool *dead_input_gpu = nullptr;  
    float *output_gpu; 
    ssize_t *argmax_input_gpu, *argmax_weight_gpu; 
    bool *dead_output_gpu;

    float *grad_output_gpu;
    float *grad_input_gpu;
    float *grad_weight_gpu;
    float *grad_bias_gpu = nullptr; 

    cudaMalloc(&input_gpu, size_input*sizeof(float)); 
    cudaMalloc(&weight_gpu, size_weight*sizeof(float)); 
    cudaMalloc(&output_gpu, size_output*sizeof(float)); 
    cudaMalloc(&argmax_input_gpu, size_output*sizeof(ssize_t)); 
    cudaMalloc(&argmax_weight_gpu, size_output*sizeof(ssize_t));
    cudaMalloc(&dead_output_gpu, size_output*sizeof(bool)); 
    cudaMalloc(&grad_output_gpu, size_output * sizeof(float)); 
    cudaMalloc(&grad_input_gpu, size_input * sizeof(float)); 
    cudaMalloc(&grad_weight_gpu, size_weight * sizeof(float)); 

    float *input_gpu_host = (float*)malloc(size_input * sizeof(float));
    float *weight_gpu_host = (float*)malloc(size_weight * sizeof(float));
    float *bias_gpu_host = nullptr; 
    bool *dead_input_gpu_host = nullptr;  
    float *output_gpu_host = (float*)malloc(size_output * sizeof(float)); 
    ssize_t *argmax_input_gpu_host = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    ssize_t *argmax_weight_gpu_host = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    bool *dead_output_gpu_host = (bool*)malloc(size_output * sizeof(bool));

    float *grad_output_gpu_host = (float*)malloc(size_output * sizeof(float)); 
    float *grad_input_gpu_host = (float*)malloc(size_input * sizeof(float));
    float *grad_weight_gpu_host = (float*)malloc(size_weight * sizeof(float));
    float *grad_bias_gpu_host = nullptr; 

    initialize(input, size_input); initialize(weight, size_weight); 

    cudaMemcpy(input_gpu, input, size_input*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(weight_gpu, weight, size_weight*sizeof(float), cudaMemcpyHostToDevice); 

    grad_output[0] = 13.0;
    std::fill(grad_input, grad_input + size_input, 0.0); 
    std::fill(grad_weight, grad_weight + size_weight, 0.0);

    cudaMemcpy(grad_output_gpu, grad_output, size_output*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(grad_input_gpu, grad_input, size_input*sizeof(float), cudaMemcpyHostToDevice); 
    cudaMemcpy(grad_weight_gpu, grad_weight, size_weight*sizeof(float), cudaMemcpyHostToDevice); 

    Timer* timer = new Timer(1); 
    morph_conv_2d_forward_cpu(input, weight, bias, dead_input, 
        K, stride, padding, B, C_in, C_out, 
        H_in, W_in, H_out, W_out, 
        output, argmax_input, argmax_weight, dead_output); 
    delete timer; 

    timer = new Timer(2);
    morph_conv_2d_backward_cpu(grad_output, argmax_input, argmax_weight, dead_output, 
        K, B, C_in, C_out, H_in, W_in, H_out, W_out, grad_input, grad_weight, grad_bias); 
    delete timer; 

    dim3 threads(THREADS_PER_BLOCK); 
    // TODO
    dim3 blocks(1); 

    timer = new Timer(3); 
    morph_conv_2d_forward_gpu<<<blocks, threads>>>(input, weight, bias, dead_input, 
        K, stride, padding, B, C_in, C_out, 
        H_in, W_in, H_out, W_out, 
        output, argmax_input, argmax_weight, dead_output); 
    delete timer; 

    timer = new Timer(4);
    morph_conv_2d_backward_gpu<<<blocks, threads>>>(grad_output, argmax_input, argmax_weight, dead_output, 
        K, B, C_in, C_out, H_in, W_in, H_out, W_out, grad_input, grad_weight, grad_bias); 
    delete timer; 

    cudaMemcpy(output_gpu_host, output_gpu, size_output*sizeof(float), cudaMemcpyDeviceToHost); 
    cudaMemcpy(argmax_input_gpu_host, argmax_input_gpu, size_output*sizeof(ssize_t), cudaMemcpyDeviceToHost); 
    cudaMemcpy(argmax_weight_gpu_host, argmax_weight_gpu, size_output*sizeof(ssize_t), cudaMemcpyDeviceToHost); 
    cudaMemcpy(grad_input_gpu_host, grad_input_gpu, size_input*sizeof(float), cudaMemcpyDeviceToHost); 
    cudaMemcpy(grad_weight_gpu_host, grad_weight_gpu, size_weight*sizeof(float), cudaMemcpyDeviceToHost);
    
    print_arr4<float>(input, B, C_in, H_in, W_in, "Input\n");
    print_arr4<float>(weight, C_out, C_in, K, K, "Weight\n");
    print_arr4<float>(output_gpu_host, B, C_out, H_out, W_out, "Output\n");
    print_arr4<ssize_t>(argmax_input_gpu_host, B, C_out, H_out, W_out, "Argmax_input\n"); 
    print_arr4<ssize_t>(argmax_weight_gpu_host, B, C_out, H_out, W_out, "Argmax_weight\n");

    print_arr4<float>(grad_input_gpu_host, B, C_in, H_in, W_in, "Grad_input\n"); 
    print_arr4<float>(grad_weight_gpu_host, C_out, C_in, K, K, "Grad_weight\n"); 

    arrequal(output, output_gpu_host, size_input); 
    arrequal(argmax_input, argmax_input_gpu_host, size_weight); 
    arrequal(argmax_weight, argmax_weight_gpu_host, size_weight); 
    arrequal(grad_input, grad_input_gpu_host, size_input); 
    arrequal(grad_weight, grad_weight_gpu_host, size_weight); 

    free(input); 
    free(weight); 
    free(output);
    free(argmax_input); 
    free(argmax_weight); 
    free(dead_output); 
    free(grad_input);
    free(grad_output); 
    free(grad_weight); 

    cudaFree(input_gpu); 
    cudaFree(weight_gpu); 
    cudaFree(output_gpu); 
    cudaFree(argmax_input); 
    cudaFree(argmax_weight);
    cudaFree(dead_output_gpu); 
    cudaFree(&grad_output_gpu); 
    cudaFree(&grad_input_gpu); 
    cudaFree(&grad_weight); 

    free(input_gpu_host); 
    free(weight_gpu_host); 
    free(output_gpu_host);
    free(argmax_input_gpu_host); 
    free(argmax_weight_gpu_host); 
    free(dead_output_gpu_host); 
    free(grad_input_gpu_host);
    free(grad_output_gpu_host); 
    free(grad_weight_gpu_host); 

    return true;
}

// More dimensions, stride, padding
bool test2() {
    int32_t B = 2, C_in = 3, C_out = 3, H_in = 5, W_in = 5, H_out = 2, W_out = 2, K = 3, stride = 3, padding = 1;
    int32_t size_input = B * C_in * H_in * W_in; 
    int32_t size_weight = C_out * C_in * K * K;
    int32_t size_output = B * C_out * H_out * W_out;
    float *input = (float*)malloc(size_input * sizeof(float)), *weight = (float*)malloc(size_weight * sizeof(float)), *bias = nullptr; bool *dead_input = nullptr;  
    float *output = (float*)malloc(size_output * sizeof(float)); 
    ssize_t *argmax_input = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    ssize_t *argmax_weight = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    bool *dead_output = (bool*)malloc(size_output * sizeof(bool));

    initialize(input, size_input); initialize(weight, size_weight); 

    morph_conv_2d_forward_cpu(input, weight, bias, dead_input, 
        K, stride, padding, B, C_in, C_out, 
        H_in, W_in, H_out, W_out, 
        output, argmax_input, argmax_weight, dead_output); 

    float *grad_output = (float*)malloc(size_output * sizeof(float)); 
    float *grad_input = (float*)malloc(size_input * sizeof(float));
    float *grad_weight = (float*)malloc(size_weight * sizeof(float));
    float *grad_bias = nullptr; 

    int curr = 1; 
    for(int i = 0; i < size_output; i++) {
        grad_output[i] = curr; 
        curr *= 2; 
        curr %= 107; 
    }
    std::fill(grad_input, grad_input + size_input, 0.0); 
    std::fill(grad_weight, grad_weight + size_weight, 0.0);

    morph_conv_2d_backward_cpu(grad_output, argmax_input, argmax_weight, dead_output, 
        K, B, C_in, C_out, H_in, W_in, H_out, W_out, grad_input, grad_weight, grad_bias); 

    print_arr4<float>(input, B, C_in, H_in, W_in, "Input\n");
    print_arr4<float>(weight, C_out, C_in, K, K, "Weight\n");
    print_arr4<float>(output, B, C_out, H_out, W_out, "Output\n");
    print_arr4<ssize_t>(argmax_input, B, C_out, H_out, W_out, "Argmax_input\n"); 
    print_arr4<ssize_t>(argmax_weight, B, C_out, H_out, W_out, "Argmax_weight\n");

    print_arr4<float>(grad_input, B, C_in, H_in, W_in, "Grad_input\n"); 
    print_arr4<float>(grad_weight, C_out, C_in, K, K, "Grad_weight\n"); 

    free(input); 
    free(weight); 
    free(output);
    free(argmax_input); 
    free(argmax_weight); 
    free(dead_output); 
    free(grad_input);
    free(grad_output); 
    free(grad_weight); 

    return true;
}

// I will use CMake with cuda. Now it is just one file. I dont think I missed any flags. i have debug on, sanitizer on, 
// optimization off (which I forgot). I will run larger tests, the last one did not catch a possible overflow. 
// I am not sure if it is possible to catch it. 

// Now I am seeing possibly a logic bug. My test was bad, I put padding = 1 on a 5x5 image instead of 4x4. 
// The convolution works fine, just one row and one column of pixels are unused. 

// Aligned padding
bool test3() {
    int32_t B = 2, C_in = 3, C_out = 3, H_in = 4, W_in = 4, H_out = 2, W_out = 2, K = 3, stride = 3, padding = 1;
    int32_t size_input = B * C_in * H_in * W_in; 
    int32_t size_weight = C_out * C_in * K * K;
    int32_t size_output = B * C_out * H_out * W_out;
    float *input = (float*)malloc(size_input * sizeof(float)), *weight = (float*)malloc(size_weight * sizeof(float)), *bias = nullptr; bool *dead_input = nullptr;  
    float *output = (float*)malloc(size_output * sizeof(float)); 
    ssize_t *argmax_input = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    ssize_t *argmax_weight = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    bool *dead_output = (bool*)malloc(size_output * sizeof(bool));

    initialize(input, size_input); initialize(weight, size_weight); 

    morph_conv_2d_forward_cpu(input, weight, bias, dead_input, 
        K, stride, padding, B, C_in, C_out, 
        H_in, W_in, H_out, W_out, 
        output, argmax_input, argmax_weight, dead_output); 

    float *grad_output = (float*)malloc(size_output * sizeof(float)); 
    float *grad_input = (float*)malloc(size_input * sizeof(float));
    float *grad_weight = (float*)malloc(size_weight * sizeof(float));
    float *grad_bias = nullptr; 

    int curr = 1; 
    for(int i = 0; i < size_output; i++) {
        grad_output[i] = curr; 
        curr *= 2; 
        curr %= 107; 
    }
    std::fill(grad_input, grad_input + size_input, 0.0); 
    std::fill(grad_weight, grad_weight + size_weight, 0.0);

    morph_conv_2d_backward_cpu(grad_output, argmax_input, argmax_weight, dead_output, 
        K, B, C_in, C_out, H_in, W_in, H_out, W_out, grad_input, grad_weight, grad_bias); 

    print_arr4<float>(input, B, C_in, H_in, W_in, "Input\n");
    print_arr4<float>(weight, C_out, C_in, K, K, "Weight\n");
    print_arr4<float>(output, B, C_out, H_out, W_out, "Output\n");
    print_arr4<ssize_t>(argmax_input, B, C_out, H_out, W_out, "Argmax_input\n"); 
    print_arr4<ssize_t>(argmax_weight, B, C_out, H_out, W_out, "Argmax_weight\n");
    print_arr4<bool>(dead_output, B, C_out, H_out, W_out, "Dead_output\n"); 

    print_arr4<float>(grad_input, B, C_in, H_in, W_in, "Grad_input\n"); 
    print_arr4<float>(grad_weight, C_out, C_in, K, K, "Grad_weight\n"); 

    free(input); 
    free(weight); 
    free(output);
    free(argmax_input); 
    free(argmax_weight); 
    free(dead_output); 
    free(grad_input);
    free(grad_output); 
    free(grad_weight); 

    return true;
}

// Bias
bool test4() {
    int32_t B = 2, C_in = 3, C_out = 3, H_in = 5, W_in = 5, H_out = 2, W_out = 2, K = 3, stride = 3, padding = 1;
    int32_t size_input = B * C_in * H_in * W_in; 
    int32_t size_weight = C_out * C_in * K * K;
    int32_t size_output = B * C_out * H_out * W_out;
    int32_t size_bias = C_out; 
    float *input = (float*)malloc(size_input * sizeof(float)), *weight = (float*)malloc(size_weight * sizeof(float)), *bias = (float*)malloc(size_bias * sizeof(float)); bool *dead_input = nullptr;  
    float *output = (float*)malloc(size_output * sizeof(float)); 
    ssize_t *argmax_input = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    ssize_t *argmax_weight = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    bool *dead_output = (bool*)malloc(size_output * sizeof(bool));

    initialize(input, size_input); initialize(weight, size_weight); 
    initialize(bias, size_bias); 

    for(int i = 0; i < size_bias; i++) bias[i] += 10.0; 

    morph_conv_2d_forward_cpu(input, weight, bias, dead_input, 
        K, stride, padding, B, C_in, C_out, 
        H_in, W_in, H_out, W_out, 
        output, argmax_input, argmax_weight, dead_output); 

    float *grad_output = (float*)malloc(size_output * sizeof(float)); 
    float *grad_input = (float*)malloc(size_input * sizeof(float));
    float *grad_weight = (float*)malloc(size_weight * sizeof(float));
    float *grad_bias = (float*)malloc(size_bias * sizeof(float)); 

    int curr = 1; 
    for(int i = 0; i < size_output; i++) {
        grad_output[i] = curr; 
        // curr *= 2; 
        // curr %= 107; 
    }
    std::fill(grad_input, grad_input + size_input, 0.0); 
    std::fill(grad_weight, grad_weight + size_weight, 0.0);
    std::fill(grad_bias, grad_bias + size_bias, 0.0); 

    morph_conv_2d_backward_cpu(grad_output, argmax_input, argmax_weight, dead_output, 
        K, B, C_in, C_out, H_in, W_in, H_out, W_out, grad_input, grad_weight, grad_bias); 

    print_arr4<float>(input, B, C_in, H_in, W_in, "Input\n");
    print_arr4<float>(weight, C_out, C_in, K, K, "Weight\n");
    print_arr4<float>(output, B, C_out, H_out, W_out, "Output\n");
    print_arr1<float>(bias, C_out, "Bias\n"); 
    print_arr4<ssize_t>(argmax_input, B, C_out, H_out, W_out, "Argmax_input\n"); 
    print_arr4<ssize_t>(argmax_weight, B, C_out, H_out, W_out, "Argmax_weight\n");

    print_arr4<float>(grad_input, B, C_in, H_in, W_in, "Grad_input\n"); 
    print_arr4<float>(grad_weight, C_out, C_in, K, K, "Grad_weight\n"); 
    print_arr1<float>(grad_bias, C_out, "Grad_bias\n"); 

    free(input); 
    free(weight); 
    free(bias);
    free(output);
    free(argmax_input); 
    free(argmax_weight); 
    free(dead_output); 
    free(grad_input);
    free(grad_output); 
    free(grad_weight); 
    free(grad_bias); 

    return true;
}

bool test5() {
    int32_t B = 32, C_in = 32, C_out = 32, H_in = 28, W_in = 28, H_out = 28, W_out = 28, K = 3, stride = 1, padding = 0;
    int32_t size_input = B * C_in * H_in * W_in; 
    int32_t size_weight = C_out * C_in * K * K;
    int32_t size_output = B * C_out * H_out * W_out;
    float *input = (float*)malloc(size_input * sizeof(float)), *weight = (float*)malloc(size_weight * sizeof(float)), *bias = nullptr; bool *dead_input = nullptr;  
    float *output = (float*)malloc(size_output * sizeof(float)); 
    ssize_t *argmax_input = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    ssize_t *argmax_weight = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    bool *dead_output = (bool*)malloc(size_output * sizeof(bool));

    initialize(input, size_input); initialize(weight, size_weight); 

    morph_conv_2d_forward_cpu(input, weight, bias, dead_input, 
        K, stride, padding, B, C_in, C_out, 
        H_in, W_in, H_out, W_out, 
        output, argmax_input, argmax_weight, dead_output); 

    float *grad_output = (float*)malloc(size_output * sizeof(float)); 
    float *grad_input = (float*)malloc(size_input * sizeof(float));
    float *grad_weight = (float*)malloc(size_weight * sizeof(float));
    float *grad_bias = nullptr; 

    grad_output[0] = 5;
    std::fill(grad_input, grad_input + size_input, 0.0); 
    std::fill(grad_weight, grad_weight + size_weight, 0.0);

    morph_conv_2d_backward_cpu(grad_output, argmax_input, argmax_weight, dead_output, 
        K, B, C_in, C_out, H_in, W_in, H_out, W_out, grad_input, grad_weight, grad_bias); 

    // print_arr4<float>(input, B, C_in, H_in, W_in, "Input\n");
    // print_arr4<float>(weight, C_out, C_in, K, K, "Weight\n");
    // print_arr4<float>(output, B, C_out, H_out, W_out, "Output\n");
    // print_arr4<ssize_t>(argmax_input, B, C_out, H_out, W_out, "Argmax_input\n"); 
    // print_arr4<ssize_t>(argmax_weight, B, C_out, H_out, W_out, "Argmax_weight\n");

    // print_arr4<float>(grad_input, B, C_in, H_in, W_in, "Grad_input\n"); 
    // print_arr4<float>(grad_weight, C_out, C_in, K, K, "Grad_weight\n"); 

    free(input); 
    free(weight); 
    free(output);
    free(argmax_input); 
    free(argmax_weight); 
    free(dead_output); 
    free(grad_input);
    free(grad_output); 
    free(grad_weight); 

    return true;
}

bool test6() {
    int32_t B = 16, C_in = 64, C_out = 64, H_in = 1024, W_in = 1024, H_out = 1024, W_out = 1024, K = 3, stride = 1, padding = 1;
    ssize_t size_input = (ssize_t)B * C_in * H_in * W_in; 
    ssize_t size_weight = (ssize_t)C_out * C_in * K * K;
    ssize_t size_output = (ssize_t)B * C_out * H_out * W_out;
    float *input = (float*)malloc(size_input * sizeof(float)), *weight = (float*)malloc(size_weight * sizeof(float)), *bias = nullptr; bool *dead_input = nullptr;  
    float *output = (float*)malloc(size_output * sizeof(float)); 
    ssize_t *argmax_input = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    ssize_t *argmax_weight = (ssize_t*)malloc(size_output * sizeof(ssize_t)); 
    bool *dead_output = (bool*)malloc(size_output * sizeof(bool));

    initialize(input, size_input); initialize(weight, size_weight); 

    morph_conv_2d_forward_cpu(input, weight, bias, dead_input, 
        K, stride, padding, B, C_in, C_out, 
        H_in, W_in, H_out, W_out, 
        output, argmax_input, argmax_weight, dead_output); 

    float *grad_output = (float*)malloc(size_output * sizeof(float)); 
    float *grad_input = (float*)malloc(size_input * sizeof(float));
    float *grad_weight = (float*)malloc(size_weight * sizeof(float));
    float *grad_bias = nullptr; 

    grad_output[0] = 5.0; 
    std::fill(grad_input, grad_input + size_input, 0.0); 
    std::fill(grad_weight, grad_weight + size_weight, 0.0);

    morph_conv_2d_backward_cpu(grad_output, argmax_input, argmax_weight, dead_output, 
        K, B, C_in, C_out, H_in, W_in, H_out, W_out, grad_input, grad_weight, grad_bias); 

    // print_arr4<float>(input, B, C_in, H_in, W_in, "Input\n");
    // print_arr4<float>(weight, C_out, C_in, K, K, "Weight\n");
    // print_arr4<float>(output, B, C_out, H_out, W_out, "Output\n");
    // print_arr4<ssize_t>(argmax_input, B, C_out, H_out, W_out, "Argmax_input\n"); 
    // print_arr4<ssize_t>(argmax_weight, B, C_out, H_out, W_out, "Argmax_weight\n");

    // print_arr4<float>(grad_input, B, C_in, H_in, W_in, "Grad_input\n"); 
    // print_arr4<float>(grad_weight, C_out, C_in, K, K, "Grad_weight\n"); 

    free(input); 
    free(weight); 
    free(output);
    free(argmax_input); 
    free(argmax_weight); 
    free(dead_output); 
    free(grad_input);
    free(grad_output); 
    free(grad_weight); 

    return true;
}

__global__ void test(int *a) {
    *a = 1;
}

int32_t main() {
    dim3 threads(1024); 
    dim3 blocks(2); 

    int a;

    Timer *timer = new Timer(1); 
    test<<<blocks, threads>>>(&a); 
    delete timer; 

    // srand(0); 
    // test1(); 
    // for(int i = 0; i < 3; i++) test1();
    // test2(); 
    // test3(); 
    // test4(); 

    // for(int i = 0; i < 10; i++) {
    //     printf("===========  %d  ===========\n", i); 
    //     auto timer = new Timer(i); 
    //     test5(); 
    //     delete timer;
    // }

    // test6(); 
    return 0; 
}
