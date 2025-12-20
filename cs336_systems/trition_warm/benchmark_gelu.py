import torch
import triton
import time
import triton.language as tl
from cuda_gelu import cuda_gelu
from typing import Callable
from statistics import mean

# 对比不同的GELU实现方式的性能
def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

def pytorch_gelu(x: torch.Tensor):
    # Use the tanh approximation to match our implementation
    return torch.nn.functional.gelu(x, approximate="tanh")

def pytorch_compiled_gelu(x: torch.Tensor):
    return torch.compile(torch.nn.functional.gelu)(x)

def cuda_gelu_func(x: torch.Tensor):
    return cuda_gelu(x)


def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()
    # Allocate output tensor
    y = torch.empty_like(x)
    # Determine grid (elements divided into blocks)
    num_elements = x.numel()
    block_size = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, block_size)
    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)
    return y

@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input is at `x_ptr` and output is at `y_ptr`
    #     |        Block 0            |          Block 1          |      ...      |
    #                            BLOCK_SIZE                                 num_elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    # Indices where this thread block should operate
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Handle boundary
    mask = offsets < num_elements
    # Read
    x = tl.load(x_ptr + offsets, mask=mask)
    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)
    # Store
    tl.store(y_ptr + offsets, y, mask=mask)
    


# ===========================================Benchmark Function===========================================
def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Time it for real now!
    times: list[float] = [] 
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()
        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        end_time = time.time()
        times.append((end_time - start_time) * 1000) 
    mean_time = mean(times) 
    return mean_time


def benchmark_gelu():
    x = torch.randn(1000, device="cuda", dtype=torch.float32)
    manual_time = benchmark("Manual GELU", lambda: manual_gelu(x))
    pytorch_time = benchmark("PyTorch GELU", lambda: pytorch_gelu(x))
    pytorch_compiled_time = benchmark("PyTorch Compiled GELU", lambda: pytorch_compiled_gelu(x))
    cuda_time = benchmark("CUDA GELU", lambda: cuda_gelu_func(x))
    triton_time = benchmark("Triton GELU", lambda: triton_gelu(x))
    print("Benchmarking GELU implementations...")
    print(f"Manual GELU: {manual_time} ms")
    print(f"PyTorch GELU: {pytorch_time} ms")
    print(f"PyTorch Compiled GELU: {pytorch_compiled_time} ms")
    print(f"CUDA GELU: {cuda_time} ms")
    print(f"Triton GELU: {triton_time} ms")

if __name__ == "__main__":
    benchmark_gelu()
