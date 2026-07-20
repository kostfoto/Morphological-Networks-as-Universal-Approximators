#include <torch/extension.h>
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

    output[output_idx] = max_val;
    argmax_idx[output_idx] = max_idx;
    argmax_weight_idx[output_idx] = best_weight_idx;
}

__global__ void maxplus_conv2d_backward_kernel(
    const int* __restrict__ argmax_idx, const int* __restrict__ argmax_weight_idx,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input, float* __restrict__ grad_weight, float* __restrict__ grad_bias,
    int B, int C_out, int H_out, int W_out) {

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

std::vector<torch::Tensor> maxplus_conv2d_forward(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int64_t stride, int64_t padding) {

    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor.");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor.");

    const auto B = input.size(0);
    const auto C_in = input.size(1);
    const auto H_in = input.size(2);
    const auto W_in = input.size(3);
    const auto C_out = weight.size(0);
    const auto K = weight.size(2);
    const auto stride_ = static_cast<int>(stride);
    const auto padding_ = static_cast<int>(padding);
    const auto H_out = (H_in + 2 * padding_ - K) / stride_ + 1;
    const auto W_out = (W_in + 2 * padding_ - K) / stride_ + 1;

    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    if (!weight.is_contiguous()) {
        weight = weight.contiguous();
    }
    if (!bias.is_contiguous()) {
        bias = bias.contiguous();
    }

    auto output = torch::full({B, C_out, H_out, W_out}, -1e9, input.options());
    auto argmax_idx = torch::full_like(output, -1, input.options().dtype(torch::kInt32));
    auto argmax_weight_idx = torch::full_like(output, -1, input.options().dtype(torch::kInt32));

    dim3 threads(BLOCK, BLOCK);
    dim3 blocks((H_out + BLOCK - 1) / BLOCK, (W_out + BLOCK - 1) / BLOCK, B * C_out);


    maxplus_conv2d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(), argmax_idx.data_ptr<int>(), argmax_weight_idx.data_ptr<int>(),
        B, C_in, C_out, H_in, W_in, H_out, W_out, K, stride_, padding_
    );

    return {output, argmax_idx, argmax_weight_idx};
}

std::vector<torch::Tensor> maxplus_conv2d_backward(
    torch::Tensor grad_output, torch::Tensor argmax_idx, torch::Tensor argmax_weight_idx,
    int64_t B, int64_t C_in, int64_t C_out,
    int64_t H_in, int64_t W_in, int64_t H_out, int64_t W_out, int64_t K) {

    auto grad_input = torch::zeros({B, C_in, H_in, W_in}, grad_output.options());
    auto grad_weight = torch::zeros({C_out, C_in, K, K}, grad_output.options());
    auto grad_bias = torch::zeros({C_out}, grad_output.options());

    dim3 threads(BLOCK, BLOCK);
    dim3 blocks((H_out + BLOCK - 1) / BLOCK, (W_out + BLOCK - 1) / BLOCK, B * C_out);

    if (!grad_output.is_contiguous()) {
        grad_output = grad_output.contiguous();
    }

    maxplus_conv2d_backward_kernel<<<blocks, threads>>>(
        argmax_idx.data_ptr<int>(), argmax_weight_idx.data_ptr<int>(),
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(), grad_weight.data_ptr<float>(), grad_bias.data_ptr<float>(),
        B, C_out, H_out, W_out
    );

    return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxplus_conv2d_forward", &maxplus_conv2d_forward, "Max-Plus Conv2D Forward (CUDA)");
    m.def("maxplus_conv2d_backward", &maxplus_conv2d_backward, "Max-Plus Conv2D Backward (CUDA)");
}


