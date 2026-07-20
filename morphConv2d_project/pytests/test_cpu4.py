import torch
import time
import copy
import random

import conv2d_thorough

# from torch.utils.cpp_extension import load
# # Load and compile CUDA extension
# print("Loading extension")
# conv2d_thorough = load(
#     name="conv2d_thorough",
#     sources=["conv2d_thorough.cu"], 
#     extra_cuda_cflags=["-std=c++17"], 
#     verbose=True
# )

from torch.autograd import Function    
class MaxPlusConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, dead_input, stride, padding):
        ctx.use_bias = True
        ctx.use_dead = True
        if bias is None: 
            ctx.use_bias = False
            bias = torch.full((weight.size(0), ), -float("inf"), dtype=input.dtype, device=input.device, requires_grad=False)
        if dead_input is None:
            ctx.use_dead = False
            dead_input = torch.zeros_like(input, dtype=torch.bool, requires_grad=False)
        output, argmax_input_idx, argmax_weight_idx, dead_output = conv2d_thorough.maxplus_conv2d_forward(
            input, weight, bias, dead_input, stride, padding
        )
        ctx.save_for_backward(argmax_input_idx, argmax_weight_idx, dead_output)
        ctx.input_shape = input.shape
        ctx.weight_shape = weight.shape
        ctx.stride = stride
        ctx.padding = padding
        return output, dead_output

    @staticmethod
    def backward(ctx, grad_output, _):
        argmax_input_idx, argmax_weight_idx, dead_output = ctx.saved_tensors
        B, C_in, H_in, W_in = ctx.input_shape
        C_out, _, K, _ = ctx.weight_shape
        stride = ctx.stride
        padding = ctx.padding
        use_bias = ctx.use_bias
        use_dead = ctx.use_dead
        H_out = (H_in + 2 * padding - K) // stride + 1
        W_out = (W_in + 2 * padding - K) // stride + 1

        grad_input, grad_weight, grad_bias = conv2d_thorough.maxplus_conv2d_backward(
            grad_output, argmax_input_idx, argmax_weight_idx, dead_output,
            C_in, H_in, W_in, K, stride, padding
        )
        return grad_input, grad_weight, grad_bias if use_bias else None, None, None, None

# Convenience wrapper
def maxplus_conv2d_wrapper(input, weight, bias=None, dead_input=None, stride=1, padding=0):
    output, dead_output = MaxPlusConv2dFunction.apply(input, weight, bias, dead_input, stride, padding)
    if dead_input is None:
        return output
    return output, dead_output
    
def check_tensor(a, b, s):
    if (a is None and b is None) or (a == b).all():
        print(f"Tensor {s} is OK.")
        return True
    else:
        print(f"Tensor {s} is not OK.")
        print((a-b).abs().max(), (a-b).abs().mean())
        return False

def test_cpu(B, C_in, C_out, 
             H_in, W_in, H_out, W_out, 
             K, stride, padding, use_bias=True, use_dead=True, use_64=False):

        working_type = torch.float64 if use_64 else torch.float32
        working_device = "cpu"

        input = torch.randn((B, C_in, H_in, W_in), dtype=working_type, device=working_device, requires_grad=True); 
        weight = torch.randn((C_out, C_in, K, K), dtype=working_type, device=working_device, requires_grad=True); 
        # Give an advantage to the bias to see its action in testing
        bias = torch.normal(5.0, 1.0, (C_out,), dtype=working_type, device=working_device, requires_grad=True) if use_bias else None 
        dead_input = torch.rand_like(input) < 0.95 if use_dead else None 
        
        start = time.time()
        ret_for = maxplus_conv2d_wrapper(input, weight, bias, dead_input, stride, padding) 
        end = time.time()
        print(f"Forward timer: {(end - start)*1000} (ms)")

        if use_dead:
            output, dead_output = ret_for
        else:
            output = ret_for
        loss = (output.max() * 13)
        start = time.time()
        loss.backward()
        end = time.time()
        print(f"Backward timer: {(end - start)*1000} (ms)")

        print("Input:\n", input)
        print("Weight:\n", weight)
        if use_bias:
            print("Bias:\n", bias)
        if use_dead:
            print("Dead input:\n", dead_input)
            print("Output:\n", output)
            print("Dead output\n:", dead_output)
        else:
            print("Output:\n", output)
        print("Grad input:\n", input.grad)
        print("Grad weight:\n", weight.grad)
        if use_bias:
            print("Grad bias:\n", bias.grad)

# More channels, resol=4x4 K=3, with stride=3, with padding=1, no bias, with dead neurons
def test_cpu4(i, use_64=False):
    print(f"===============  Test 4.{i} (start)  ================\n")
    B = 1; C_in = 3; C_out = 3; H_in = 4; W_in = 4; H_out = 2; W_out = 2; K = 3; stride = 3; padding = 1
    
    test_cpu(B, C_in, C_out, H_in, W_in, H_out, W_out, K, stride, padding, False, True, use_64)
    print(f"================  Test 4.{i} (end)  =================\n")

def __main__():
    random.seed(0); 
    torch.manual_seed(0); 

    print("CUDA availability:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())

    for i in range(3):
        test_cpu4(i+1); 
    for i in range(3):
        test_cpu4(i+1+3, True); 

__main__()