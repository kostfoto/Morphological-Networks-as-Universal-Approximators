#include <bits/stdc++.h>
#include <string>

// #include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK 16  // TILE SIZE

__global__ void maxplus_conv2d_forward_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int* __restrict__ argmax_idx, int* __restrict__ argmax_weight_idx,
    int B, int C_in, int C_out,
    int H_in, int W_in, int H_out, int W_out,
    int K, int stride, int padding) {
    
    printf("Hello words\n"); 

    // int b = blockIdx.z;
    // int oc = blockIdx.y;
    // int i = blockIdx.x * blockDim.y + threadIdx.y;
    // int j = threadIdx.x;

    // if (i >= H_out || j >= W_out) return;

    int i = blockIdx.x * BLOCK + threadIdx.x;
    int j = blockIdx.y * BLOCK + threadIdx.y;
    if (i >= H_out || j >= W_out) return;

    int b = blockIdx.z / C_out;
    int oc = blockIdx.z % C_out;

    float max_val = -1e9;
    int max_idx = -1;
    int best_weight_idx = -1;

    for (int ic = 0; ic < C_in; ++ic) {
        for (int ki = 0; ki < K; ++ki) {
            for (int kj = 0; kj < K; ++kj) {
                int ii = i * stride - padding + ki;
                int jj = j * stride - padding + kj;

                if (ii >= 0 && ii < H_in && jj >= 0 && jj < W_in) {
                    int input_idx = ((b * C_in + ic) * H_in + ii) * W_in + jj;
                    int weight_idx = ((oc * C_in + ic) * K + ki) * K + kj;
                    float val = input[input_idx] + weight[weight_idx];

                    if (val > max_val) {
                        max_val = val;
                        max_idx = input_idx;
                        best_weight_idx = weight_idx;
                    }
                }
            }
        }
    }

    int output_idx = ((b * C_out + oc) * H_out + i) * W_out + j;

    if (bias != nullptr && bias[oc] > max_val) {
        max_val = bias[oc];
        max_idx = -1;
        best_weight_idx = -1;
    }
    
    printf("%.3f", max_val); 

    output[output_idx] = max_val;
    argmax_idx[output_idx] = max_idx;
    argmax_weight_idx[output_idx] = best_weight_idx;
}

__global__ void maxplus_conv2d_backward_kernel(
    const int* __restrict__ argmax_idx, const int* __restrict__ argmax_weight_idx,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input, float* __restrict__ grad_weight, float* __restrict__ grad_bias,
    int B, int C_out, int H_out, int W_out) {

    // int b = blockIdx.z;
    // int oc = blockIdx.y;
    // int i = blockIdx.x * blockDim.y + threadIdx.y;
    // int j = threadIdx.x;

    // if (i >= H_out || j >= W_out) return;

    int i = blockIdx.x * BLOCK + threadIdx.x;
    int j = blockIdx.y * BLOCK + threadIdx.y;
    if (i >= H_out || j >= W_out) return;

    int b = blockIdx.z / C_out;
    int oc = blockIdx.z % C_out;

    int output_idx = ((b * C_out + oc) * H_out + i) * W_out + j;
    int max_idx = argmax_idx[output_idx];
    int weight_idx = argmax_weight_idx[output_idx];
    float go = grad_output[output_idx];

    if (max_idx >= 0) {
        atomicAdd(&grad_input[max_idx], go);
    }
    if (weight_idx >= 0) {
        atomicAdd(&grad_weight[weight_idx], go);
    }
    if (max_idx == -1 && weight_idx == -1) {
        atomicAdd(&grad_bias[oc], go);
    }
}

template <typename T>
void printarr(T *a, int n, const char* s = "") {
	std::cout << s << std::endl; 
	for(int i = 0; i < n; i++) std::cout << a[i] << ", "; std::cout << std::endl;  
}

template <typename T>
void initialize(T* a, int n) {
	for(int i = 0; i < n; i++) {
		a[i] = 1.0 * (rand() % 1000) / 100;
	}
}

void test() {
	int B = 1, C_in = 1, C_out = 1, H_in = 3, W_in = 3, H_out = 1, W_out = 1, K = 3, stride = 1, padding = 0; 
	
	int size_input = B*C_in*H_in*W_in, size_weight = C_out*C_in*K*K, size_bias = C_out, size_output = B*C_out*H_out*W_out; 
	
	float *input = (float*)malloc(size_input*sizeof(float)); 
	float *weight = (float*)malloc(size_weight*sizeof(float));
	float *bias = (float*)malloc(size_bias*sizeof(float));
	float *output = (float*)malloc(size_output*sizeof(float));
	int *argmax_input = (int*)malloc(size_output*sizeof(int)); 
	int *argmax_weight = (int*)malloc(size_output*sizeof(int)); 
	
	float *grad_output = (float*)malloc(size_output*sizeof(float)); 
	float *grad_input = (float*)malloc(size_input*sizeof(float)); 
	float *grad_weight = (float*)malloc(size_weight*sizeof(float)); 
	float *grad_bias = (float*)malloc(size_bias*sizeof(float)); 
	
	initialize(input, size_input); initialize(weight, size_weight); initialize(grad_output, size_output);
	for(int i = 0; i < size_bias; i++) bias[i] = -1e9; 
	for(int i = 0; i < size_output; i++) output[i] = -1e9; 
	for(int i = 0; i < size_input; i++) grad_input[i] = 0.0; 
	for(int i = 0; i < size_weight; i++) grad_weight[i] = 0.0;
	
	float *input_gpu; cudaMalloc(&input_gpu, size_input*sizeof(float)); 
	float *weight_gpu; cudaMalloc(&weight_gpu, size_weight*sizeof(float));
	float *bias_gpu; cudaMalloc(&bias_gpu, size_bias*sizeof(float));
	float *output_gpu; cudaMalloc(&output_gpu, size_output*sizeof(float));
	int *argmax_input_gpu; cudaMalloc(&argmax_input_gpu, size_output*sizeof(int)); 
	int *argmax_weight_gpu; cudaMalloc(&argmax_weight_gpu, size_output*sizeof(int)); 
	
	float *grad_output_gpu; cudaMalloc(&grad_output_gpu, size_output*sizeof(float)); 
	float *grad_input_gpu; cudaMalloc(&grad_input_gpu, size_input*sizeof(float)); 
	float *grad_weight_gpu; cudaMalloc(&grad_weight_gpu, size_weight*sizeof(float)); 
	float *grad_bias_gpu; cudaMalloc(&grad_bias_gpu, size_bias*sizeof(float)); 
	
	cudaMemcpy(input_gpu, input, size_input, cudaMemcpyHostToDevice);
	cudaMemcpy(weight_gpu, weight, size_weight, cudaMemcpyHostToDevice);
	cudaMemcpy(bias_gpu, bias, size_bias, cudaMemcpyHostToDevice);
	cudaMemcpy(output_gpu, output, size_output, cudaMemcpyHostToDevice);
	cudaMemcpy(grad_output_gpu, grad_output, size_output, cudaMemcpyHostToDevice);
	cudaMemcpy(grad_input_gpu, grad_input, size_input, cudaMemcpyHostToDevice);
	cudaMemcpy(grad_weight_gpu, grad_weight, size_weight, cudaMemcpyHostToDevice);
	
	// dim3 threads(BLOCKS, BLOCKS);
	// dim3 blocks((H_out + BLOCK - 1) / BLOCK, (W_out + BLOCK - 1) / BLOCK, B * C_out); 
	
	dim3 threads(1, 1); 
	dim3 blocks(1, 1); 
	
	printf("%d\n", (H_out + BLOCK - 1) / BLOCK); 
	
	maxplus_conv2d_forward_kernel<<<blocks, threads>>>(
		input_gpu, weight_gpu, bias_gpu, 
		output_gpu, argmax_input_gpu, argmax_weight_gpu, 
		B, C_in, C_out, H_in, W_in, H_out, W_out, K, stride, padding
	); 
	
	maxplus_conv2d_backward_kernel<<<blocks, threads>>>(
		argmax_input_gpu, argmax_weight_gpu, grad_output_gpu, 
		grad_input_gpu, grad_weight_gpu, grad_bias_gpu, 
		B, C_out, H_out, W_out
	);
	
	cudaMemcpy(input, input_gpu, size_input, cudaMemcpyDeviceToHost);
	cudaMemcpy(weight, weight_gpu, size_weight, cudaMemcpyDeviceToHost);
	cudaMemcpy(bias, bias_gpu, size_bias, cudaMemcpyDeviceToHost);
	cudaMemcpy(output, output_gpu, size_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(argmax_weight, argmax_weight_gpu, size_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(argmax_input, argmax_input_gpu, size_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(grad_output, grad_output_gpu, size_output, cudaMemcpyDeviceToHost);
	cudaMemcpy(grad_input, grad_input_gpu, size_input, cudaMemcpyDeviceToHost);
	cudaMemcpy(grad_weight, grad_weight_gpu, size_weight, cudaMemcpyDeviceToHost);
	
	printarr(input, 9, "Input");
	printarr(weight, 9, "Weight");
	printarr(bias, 1, "Bias");
	printarr(output, 1, "Output");
	printarr(argmax_input, 1, "argmax_input");
	printarr(argmax_weight, 1, "argmax_weight"); 
	printarr(grad_output, 1, "Grad_output"); 
	printarr(grad_input, 9, "Grad_input"); 
	printarr(grad_weight, 9, "Grad_weight"); 
	
	cudaFree(input_gpu); 
	cudaFree(weight_gpu); 
	cudaFree(bias_gpu); 
	cudaFree(output_gpu);
	cudaFree(argmax_weight_gpu);
	cudaFree(argmax_input_gpu);
	cudaFree(grad_output_gpu); 
	cudaFree(grad_input_gpu); 
	cudaFree(grad_weight_gpu);
}

int main() {
	for(int i = 0; i < 1; i++) {
		test(); 
	}
	return 0;
}
