import torch
import time
import copy
import random

import conv2d_cuda
import conv2d_thorough

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
def maxplus_conv2d_wrapper(input, weight, bias=None, stride=1, padding=0, dead_input=None):
    output, dead_output = MaxPlusConv2dFunction.apply(input, weight, bias, dead_input, stride, padding)
    if dead_input is None:
        return output
    return output, dead_output

from torch.autograd import Function    
class MaxPlusConv2dFunctionOriginal(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        output, argmax_input_idx, argmax_weight_idx = conv2d_cuda.maxplus_conv2d_forward(
            input, weight, bias, stride, padding
        )
        ctx.save_for_backward(argmax_input_idx, argmax_weight_idx)
        ctx.input_shape = input.shape
        ctx.weight_shape = weight.shape
        ctx.stride = stride
        ctx.padding = padding
        return output

    @staticmethod
    def backward(ctx, grad_output):
        argmax_input_idx, argmax_weight_idx = ctx.saved_tensors
        B, C_in, H_in, W_in = ctx.input_shape
        C_out, _, K, _ = ctx.weight_shape
        stride = ctx.stride
        padding = ctx.padding
        H_out = (H_in + 2 * padding - K) // stride + 1
        W_out = (W_in + 2 * padding - K) // stride + 1

        grad_input, grad_weight, grad_bias = conv2d_cuda.maxplus_conv2d_backward(
            grad_output, argmax_input_idx, argmax_weight_idx,
            B, C_in, C_out, H_in, W_in, H_out, W_out, K
        )
        return grad_input, grad_weight, grad_bias, None, None

# Convenience wrapper
def maxplus_conv2d_wrapper_original(input, weight, bias=None, stride=1, padding=0):
    return MaxPlusConv2dFunctionOriginal.apply(input, weight, bias, stride, padding)

    
def check_tensor(a, b, s):
    if (a is None and b is None) or (a == b).all():
        print(f"Tensor {s} is OK.")
        return True
    else:
        print(f"Tensor {s} is not OK.")
        print((a-b).abs().max(), (a-b).abs().mean())
        return False

def test_gpu(B, C_in, C_out, 
             H_in, W_in, H_out, W_out, 
             K, stride, padding, use_bias=True, use_64=False):

        assert(torch.cuda.is_available())
        
        working_type = torch.float64 if use_64 else torch.float32
        working_device = "cuda"

        input = torch.randn((B, C_in, H_in, W_in), dtype=working_type, device=working_device, requires_grad=True); 
        weight = torch.randn((C_out, C_in, K, K), dtype=working_type, device=working_device, requires_grad=True); 
        # Give an advantage to the bias to see its action in testing
        bias = torch.normal(5.0, 1.0, (C_out,), dtype=working_type, device=working_device, requires_grad=True) if use_bias else None 
        
        start = time.time()
        ret_for = maxplus_conv2d_wrapper_original(input, weight, bias, stride, padding)
        end = time.time()
        print(f"Forward timer (GPU): {(end - start)*1000} (ms)")

        output = ret_for

        # print(output.isinf().sum())

        rand_out_weight = torch.randn_like(output)
        
        loss = (output * rand_out_weight).max()
        start = time.time()
        loss.backward()
        end = time.time()
        print(f"Backward timer (GPU): {(end - start)*1000} (ms)")

        input_cpu = input.to("cpu").detach().requires_grad_(True)
        weight_cpu = weight.to("cpu").detach().requires_grad_(True)
        bias_cpu = bias.to("cpu").detach().requires_grad_(True) if use_bias else None
        dead_input_cpu = None

        start = time.time()
        ret_for_cpu = maxplus_conv2d_wrapper(input_cpu, weight_cpu, bias_cpu, stride, padding, dead_input_cpu,)
        end = time.time()
        print(f"Forward timer (CPU): {(end - start)*1000} (ms)")
        
        output_cpu = ret_for_cpu

        # print(output_cpu.isinf().sum())

        loss_cpu = (output_cpu * rand_out_weight.to("cpu")).max()
        start = time.time()
        loss_cpu.backward()
        end = time.time()
        print(f"Backward timer (CPU): {(end - start)*1000} (ms)")

        ok = True
        ok = False if not check_tensor(output.to("cpu"), output_cpu, "Output") else ok
        ok = False if not check_tensor(input.grad.to("cpu"), input_cpu.grad, "Grad input") else ok
        ok = False if not check_tensor(weight.grad.to("cpu"), weight_cpu.grad, "Grad weight") else ok
        if use_bias:
            ok = False if not check_tensor(bias.grad.to("cpu"), bias_cpu.grad, "Grad bias") else ok
        
        # print((weight.max() + input.max()))

        # print(output)
        # print(output_cpu)

        if ok:
            print("Test OK")
        else:
            print("Test NOT OK")
            exit()

#  Stress testing. Realistic matrices, compared against the gpu results
def test_gpu5(i, use_64=False):
    print(f"===============  Test 5.{i} (start)  ================\n")
    B = 8; C_in = 256; C_out = 256; H_in = 16; W_in = 16; H_out = 16; W_out = 16; K = 5; stride = 1; padding = 2
    
    test_gpu(B, C_in, C_out, H_in, W_in, H_out, W_out, K, stride, padding, True, use_64)

    B = 8; C_in = 16; C_out = 16; H_in = 256; W_in = 256; H_out = 256; W_out = 256; K = 3; stride = 1; padding = 1
    
    test_gpu(B, C_in, C_out, H_in, W_in, H_out, W_out, K, stride, padding, True, use_64)
    print(f"================  Test 5.{i} (end)  =================\n")


def __main__():
    random.seed(0); 
    torch.manual_seed(0); 

    print("CUDA availability:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())

    for i in range(6):
        test_gpu5(i+1); 

__main__()