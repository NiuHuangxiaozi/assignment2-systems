import torch
import triton
import triton.language as tl
from typing import Any



# from local code
from flash_attention_torch import NIUFlashAttentionV2_torch
from flash_attention_triton import NiuFlashAttentionV2_triton


# ————————————————————————————————
# Benchmark 配置
# ————————————————————————————————

# 全局定死
B = 1  # batchsize
DTYPES = [torch.float16, torch.bfloat16, torch.float32]



# 探究N_CTX对于flashattentionV2的影响，对应于bench_attention_N_CTX函数
Default_D = 16
N_CTX = [2**i for i in range(7,12)]  # 128 (2^7) 到 65536 (2^16)


# 探究D对于flashattentionV2的影响，对应于bench_attention_D函数
Default_N_CTX = 128
D = [16, 64, 128]



# 或者你有一个封装好的 PyTorch 函数，如：
def attention_forward(Q, K, V, causal=True):
    # 这里可以调用你的 Triton kernel 或 torch.nn.functional.scaled_dot_product_attention
    # 示例（仅用于 benchmark 结构，非真实 kernel）：
    attn = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    return attn


# 计算attention的flops
def get_attention_forward_function(B, N_CTX, D)-> int:
    # Q @ K.transpose(-2, -1)* scale的计算量
    Computational_complexity_1 = B * 2  * N_CTX  * D * N_CTX + B * N_CTX * N_CTX
    # softmax的计算量
    Computational_complexity_2 = 3 * B * N_CTX * N_CTX + B * N_CTX * (N_CTX-1)
    # O @ V的计算量
    Computational_complexity_3 = B * N_CTX * N_CTX * D
    # 总的计算量
    Total_Computational_complexity = Computational_complexity_1 + Computational_complexity_2 + Computational_complexity_3
    return Total_Computational_complexity





# forward

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N_CTX"],  # x-axis: sequence length
        x_vals=N_CTX,
        line_arg="provider",   # different lines for different head/embedding dims
        line_vals=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        line_names=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="TFLOPS",     # 或 "ms" if using `do_bench` with timing
        plot_name="flashattentionV2-forward-N_CTX-performance comparison",
        args={"dtype": torch.float32, "D": Default_D, "B": 1},  # 默认 dtype；会通过多个 report 覆盖所有 dtypes
    )
)
def bench_attention_N_CTX(D, N_CTX, B, dtype=torch.bfloat16, provider="standard_attention"):
    
    q = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype)
    k = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype)
    v = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "standard_attention":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: attention_forward(q, k, v, causal=True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: NiuFlashAttentionV2_triton.apply(q, k, v, True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: NIUFlashAttentionV2_torch.apply(q, k, v, True),
            quantiles=quantiles
        )
    # 这里计算TFLOPS
    perf = lambda ms: get_attention_forward_function(B, N_CTX, D) / (ms * 1e-3) / 1e12
    return perf(ms), perf(min_ms), perf(max_ms)
 


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["D"],  # x-axis: sequence length
        x_vals=D,
        line_arg="provider",   # different lines for different head/embedding dims
        line_vals=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        line_names=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="TFLOPS",     # 或 "ms" if using `do_bench` with timing
        plot_name="flashattentionV2-forward-D-performance comparison",
        args={"dtype": torch.float32, "N_CTX": Default_N_CTX, "B": 1},  # 默认 dtype；会通过多个 report 覆盖所有 dtypes
    )
)
def bench_attention_D(N_CTX, D, B, dtype=torch.bfloat16, provider="standard_attention"):
    
    q = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype)
    k = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype)
    v = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "standard_attention":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: attention_forward(q, k, v, causal=True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            
            lambda: NiuFlashAttentionV2_triton.apply(q, k, v, True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: NIUFlashAttentionV2_torch.apply(q, k, v, True),
            quantiles=quantiles
        )

    # 这里计算TFLOPS
    perf = lambda ms: get_attention_forward_function(B, N_CTX, D) / (ms * 1e-3) / 1e12
    return perf(ms), perf(min_ms), perf(max_ms)






# backward

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["D"],  # x-axis: sequence length
        x_vals=D,
        line_arg="provider",   # different lines for different head/embedding dims
        line_vals=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        line_names=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="TFLOPS",     # 或 "ms" if using `do_bench` with timing
        plot_name="flashattentionV2-backward-D-performance comparison",
        args={"dtype": torch.float32, "N_CTX": Default_N_CTX, "B": 1},  # 默认 dtype；会通过多个 report 覆盖所有 dtypes
    )
)
def bench_attention_backward_D(N_CTX, D, B, dtype=torch.bfloat16, provider="standard_attention"):

    # 定义需要梯度的叶子节点
    q = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == "standard_attention":
        standard_attention_output = attention_forward(q, k, v,True)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: standard_attention_output.sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_triton":
        flashattentionV2_triton_output = NiuFlashAttentionV2_triton.apply(q, k, v, True)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flashattentionV2_triton_output.sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_torch":
        flashattentionV2_torch_output = NIUFlashAttentionV2_torch.apply(q, k, v, True)      
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flashattentionV2_torch_output.sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")

    perf = lambda ms: 2 * get_attention_forward_function(B, N_CTX, D) / (ms * 1e-3) / 1e12
    return perf(ms), perf(min_ms), perf(max_ms)





# 我们研究N_CTX对于flashattentionV2的影响，对应于bench_attention_backward_N_CTX函数
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N_CTX"],  # x-axis: sequence length
        x_vals=N_CTX,
        line_arg="provider",   # different lines for different head/embedding dims
        line_vals=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        line_names=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel="TFLOPS",     # 或 "ms" if using `do_bench` with timing
        plot_name="flashattentionV2-backward-N_CTX-performance comparison",
        args={"dtype": torch.float32, "D": Default_D, "B": 1},  # 默认 dtype；会通过多个 report 覆盖所有 dtypes
    )
)
def bench_attention_backward_N_CTX(D, N_CTX, B, dtype=torch.bfloat16, provider="standard_attention"):
    
    q = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == "standard_attention":
        standard_attention_output = attention_forward(q, k, v, True)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: standard_attention_output.sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_triton":
        flashattentionV2_triton_output = NiuFlashAttentionV2_triton.apply(q, k, v, True)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flashattentionV2_triton_output.sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_torch":
        flashattentionV2_torch_output = NIUFlashAttentionV2_torch.apply(q, k, v, True)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flashattentionV2_torch_output.sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")

    perf = lambda ms: 2 * get_attention_forward_function(B, N_CTX, D) / (ms * 1e-3) / 1e12
    return perf(ms), perf(min_ms), perf(max_ms)




# +++++++++++++++++++++++++++
# End2End
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N_CTX"],  # x-axis: sequence length
        x_vals=N_CTX,
        line_arg="provider",   # different lines for different head/embedding dims
        line_vals=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        line_names=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        ylabel="TFLOPS",     # 或 "ms" if using `do_bench` with timing
        plot_name="flashattentionV2-end2end-N_CTX-performance comparison",
        args={"dtype": torch.float32, "D": Default_D, "B": 1},  # 默认 dtype；会通过多个 report 覆盖所有 dtypes
    )
)
def benchmark_end2end_N_CTX(D, N_CTX, B, dtype=torch.bfloat16, provider="standard_attention"):
    
    q = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == "standard_attention":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: attention_forward(q, k, v, True).sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: NiuFlashAttentionV2_triton.apply(q, k, v, True).sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: NIUFlashAttentionV2_torch.apply(q, k, v, True).sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")

    perf = lambda ms: 3 * get_attention_forward_function(B, N_CTX, D) / (ms * 1e-3) / 1e12
    return perf(ms), perf(min_ms), perf(max_ms)



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["D"],  # x-axis: sequence length
        x_vals=D,
        line_arg="provider",   # different lines for different head/embedding dims
        line_vals=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        line_names=["standard_attention", "flashAttentionV2_triton", "flashAttentionV2_torch"],
        ylabel="TFLOPS",     # 或 "ms" if using `do_bench` with timing
        plot_name="flashattentionV2-end2end-D-performance comparison",
        args={"dtype": torch.float32, "N_CTX": Default_N_CTX, "B": 1},  # 默认 dtype；会通过多个 report 覆盖所有 dtypes
    )
)
def benchmark_end2end_D(N_CTX, D, B, dtype=torch.bfloat16, provider="standard_attention"):
    
    q = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B,  N_CTX, D, device="cuda", dtype=dtype, requires_grad=True)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == "standard_attention":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: attention_forward(q, k, v, True).sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: NiuFlashAttentionV2_triton.apply(q, k, v, True).sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    elif provider == "flashAttentionV2_torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: NIUFlashAttentionV2_torch.apply(q, k, v, True).sum().backward(retain_graph=True),
            quantiles=quantiles
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")

    perf = lambda ms: 3 * get_attention_forward_function(B, N_CTX, D) / (ms * 1e-3) / 1e12
    return perf(ms), perf(min_ms), perf(max_ms)


import argparse
import logging
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward", action="store_true", default=False)
    parser.add_argument("--backward", action="store_true", default=False)
    parser.add_argument("--forward_backward", action="store_true", default=False)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    
    
    logging.info("Benchmarking...")
    if args.forward:
        logging.info("Benchmarking forward pass...")
        for dtype in DTYPES:
            # 探究D对于flashattentionV2的影响
            bench_attention_D.run(
                print_data=True,
                save_path=f"./benchmark_result/benchmark_causal_b1_N_CTX_{Default_N_CTX}_{str(dtype).split('.')[-1]}",
                show_plots=True
            )
            
            
            # 探究N_CTX对于flashattentionV2的影响
            bench_attention_N_CTX.run(
                print_data=True,
                save_path=f"./benchmark_result/benchmark_causal_b1_D{Default_D}_{str(dtype).split('.')[-1]}",
                show_plots=True
            )
        logging.info("Forward pass benchmarking completed.")
    
        
    if args.backward:
        logging.info("Benchmarking backward pass...")
        # 测试反向传播
        for dtype in DTYPES:
            bench_attention_backward_D.run(
                print_data=True,
                save_path=f"./benchmark_result/benchmark_causal_b1_N_CTX{Default_N_CTX}_{str(dtype).split('.')[-1]}",
                show_plots=True
            )
            
            bench_attention_backward_N_CTX.run(
                print_data=True,
                save_path=f"./benchmark_result/benchmark_causal_b1_D{Default_D}_{str(dtype).split('.')[-1]}",
                show_plots=True
            )
        logging.info("Backward pass benchmarking completed.")
    
    if args.forward_backward:
        logging.info("Benchmarking forward and backward pass...")
        for dtype in DTYPES:
            benchmark_end2end_N_CTX.run(
                print_data=True,
                save_path=f"./benchmark_result/benchmark_causal_b1_D{Default_D}_{str(dtype).split('.')[-1]}",
                show_plots=True
            )
            
            benchmark_end2end_D.run(
                print_data=True,
                save_path=f"./benchmark_result/benchmark_causal_b1_N_CTX{Default_N_CTX}_{str(dtype).split('.')[-1]}",
                show_plots=True
            )
        logging.info("Forward and backward pass benchmarking completed.")
    logging.info("Benchmarking completed.")
        