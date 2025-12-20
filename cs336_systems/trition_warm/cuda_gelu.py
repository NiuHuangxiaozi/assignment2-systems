import os
import torch
from torch.utils.cpp_extension import load_inline

# === Step 1: 设置 CUDA 同步模式（出错时立刻报错，方便调试）===
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



'''
CUDA代码的解释：
__global__：告诉编译器这是个 GPU kernel 函数，只能从主机（CPU）调用，在设备（GPU）上运行。
TORCH_CHECK(...)：类似 assert，不满足就抛异常，提示信息超友好。
gelu_kernel<<<...>>>：启动 CUDA kernel！<<<num_blocks, block_size>>> 是 CUDA 的 launch 语法。
C10_CUDA_KERNEL_LAUNCH_CHECK()：如果 kernel 启动失败（比如 block 太大、grid 太大），这里会立刻报错！

'''
# === Step 2: 写 CUDA 核心代码（GELU 算子）===
cuda_gelu_src = r'''
#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

__global__ void gelu_kernel(float* in, float* out, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // sqrt(2/pi) ≈ 0.79788456
        float x = in[i];
        out[i] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

torch::Tensor gelu(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    torch::Tensor y = torch::empty_like(x);
    int num_elements = x.numel();
    int block_size = 1024;
    int num_blocks = cdiv(num_elements, block_size);
    
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK(); // 检查 kernel 是否启动成功
    return y;
}
'''

# === Step 3: C++ 头文件声明（告诉 Python 有一个叫 gelu 的函数）===
# Python 通过这个“签名”生成 Python 绑定。内部用 PyBind11 生成 Python 接口，它需要知道 C++ 函数长啥样。 
cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

# === Step 4: 编译并加载 CUDA 算子 ===
print("正在编译 CUDA GELU 算子...")


# 👇 确保 build_directory 存在
os.makedirs("./var/cuda_gelu", exist_ok=True)


'''
为什么需要分开写？
因为 PyBind11（Python 绑定工具）只认识 C++ 函数声明，但它不需要知道 CUDA 实现细节。

cpp_sources：给 PyBind11 看的接口契约。
cuda_sources：给 nvcc 看的实现代码（包含 GPU kernel）。
虽然你的 cuda_sources 里已经包含了 gelu 函数的完整定义，但 load_inline 仍然要求你通过 cpp_sources 明确声明哪些函数要暴露给 Python —— 这是一种显式约定，避免歧义。

🤯 想象一下：你把整个菜谱（包括采购、切菜、炒菜）写在一张纸上，但为了让服务员（Python）知道“今天有宫保鸡丁”，你还是得在菜单（cpp_sources）上单独写一行：“宫保鸡丁 —— 有”。 
'''
module = load_inline(
    name="inline_gelu",
    cpp_sources=cpp_gelu_src,
    cuda_sources=cuda_gelu_src,
    functions=["gelu"],
    extra_cflags=["-O2"],
    extra_cuda_cflags=["-O2"],
    build_directory="./var/cuda_gelu",  # 编译产物放这里，下次运行更快
    verbose=True
)
cuda_gelu = module.gelu

# === Step 5: 测试算子 ===
if __name__ == "__main__":
    # 创建一个随机的 CUDA 张量
    x = torch.randn(10000, device="cuda", dtype=torch.float32)
    
    # 用你写的 CUDA 算子计算 GELU
    y_custom = cuda_gelu(x)
    
    # 用 PyTorch 官方 GELU 做对比
    y_torch = torch.nn.functional.gelu(x)
    
    # 检查结果是否一致（允许微小误差）
    max_diff = (y_custom - y_torch).abs().max().item()
    print(f"最大误差: {max_diff:.1f}")
    
    if max_diff < 1e-3:
        print("✅ 成功！你的 CUDA GELU 算子和 PyTorch 结果一致！")
    else:
        print("❌ 结果不一致，请检查实现。")