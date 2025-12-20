


# 1、Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

答：Flops不能完全代表计算量，因为机器运行的时候不是一直在算算算，还有io、同步异步等其他的操作，如果我们的Q矩阵和K矩阵是M *M 
所以
$$
MatrixMultiply\_Flops = 2 M ^3
$$
而softmax每一个元素主要进行取指数，求和加法和除法三个操作，flops可以简写为：
$$
MatrixMultiply\_Flops = 3 L * L
$$
我在实验的时候，M是512，L是256，所以倍数是：
$$
    2 * 512^3 / 3 * 256 * 256 = 1365倍
$$
但是实际用nsys profile测出来的时间倍数是2倍多一点。



# 二、Profile running one complete training step with your implementation of AdamW (i.e., the forwardpass, computing the loss and running a backward pass, and finally an optimizer step, as you’d do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels

![实验对比](./statics/nsys_forward_total.png)
里面算子最多的就是 ampere_sgemm_128x64_tn 分块矩阵的乘法应该是用到这个 128 * 64的小矩阵的乘法了，tn表示前面的转置了，后面的没有转置（有可能相反，猜想可能是qt^T）两个过程差不多


# 三、Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?
```
  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
  void at::native::vectorized_elementwise_kernel<(int)4, at::native::exp_kernel_cuda(at::TensorIterat…
```
上面三个kernel 函数是出了128* 64之外调用最多的函数，他们都是c++函数。
elementwise_kernel表示的是逐元素的操作, 隶属于at（Aten命名空间下native子命名空间）的一个算子函数，这是一个模板，我们可以输入不同的参数控制并行的规模，（int)128表示的是启动128个线程，（int）表示每一个线程处理两个元素，后面是对每一个元素的具体操作（学过c++模板函数应该非常熟悉）。

vectorized_elementwise_kernel是处理每一个向量，比如可以一次性处理4到8个元素这样子的，它执行的函数是exp_kernel_cuda，很有可能是softmax中间需要的。

# 四、What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the “CUDA GPU Kernel Summary” under “Stats Systems View”, and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)
可以看到都是ampere_sgemm_128x64_tn算子，这个矩阵乘法算子占了大头


# 五、You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?
layernorm 应该是可以用bf16的，毕竟我想了一下，layernorm里面一共三个操作，求和，减法和除法。bf16和fp32表示的范围是一样的，所以加法和除法没问题，减法得到的值肯定比梯度大，出问题的概率较小，但查了许多资料，还是强制使用fp32保证训练的稳定性。

# 六、Modify your benchmarking script to optionally run the model using mixed precision with BF16. Time the forward and backward passes with and without mixed-precision for each language model size described in §1.1.2. Compare the results of using full vs. mixed precision, and comment on any trends as model size changes. You may find the nullcontext no-op context manager to be useful

![一次前向后向运行时间](./statics/mixprecision.png)

```python
from contextlib import nullcontext
cm = autocast(device='cuda', dtype=torch.bfloat16) if use_mixed_precision else nullcontext()
with cm:
    # run model forward/backward

```

# 七、Add an option to your profiling script to run your model through the memory profiler. It may be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load specific model sizes, etc). Then, run your script to get a memory profile of the 2.7B model when either doing inference only (just forward pass) or a full training step. How do your memory timelines look like? Can you tell which stage is running based on the peaks you see?
我们观察到的样子是：我做了三次前向传播，这个图就显示了三次的山峰；反向传播也大概是三座山峰，但是许多中间变量的在显存上的时间变长了（胖的山）
随着transformerLM第51行（对于我的代码） output = self.lm_head(x)的结束，显存分配出现了几个快速峰值(山峰), 然后就开始反向传播了

# 八、What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?
我使用了下面的模型：
```
model = NiuTransformerLM(10000,
                                256,
                                512,
                                4,
                                16,
                                1344,
                                10000,
                                device="cuda")
dummy_data = torch.randint(0, 10000, (10, args.context_length)).to("cuda")
```
分别对于输入context_length =100，150和200做了实验：得到下面的表格：
| 长度   | 前向显存 | 整体显存 |
|--------|------|----------|
| 100    | 420M   |   675M |
| 150    | 625M   |   935M |
| 200    | 890M   |   1.2G  |


# 九、Find the peak memory usage of the 2.7B model when using mixed-precision, for both a forward pass and a full optimizer step. Does mixed-precision significantly affect memory usage?

在整体训练中，使用混合精度，最大显存占比从 1.2G 降为 975M；
对于前向传播，最大显存从 890M降为 690M， 确实能够降低显存占用

# 十、Consider the 2.7B model. At our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MB (i.e.,divide the number of bytes by 10242).
感觉就是计算ffn和atten的激活值的大小。
这里计算十分的复杂，但是方法是一样的，这里我们就以ffn模块为例子，计算一下激活值的大小：
模块的主要结构如下：
```python
class NIUSWIGLUFFN(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device = None, dtype = None):
        super(NIUSWIGLUFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        
        self.w1 = NIULinear(d_model, d_ff, device = device, dtype = dtype)
        self.w2 = NIULinear(d_ff, d_model, device = device, dtype = dtype)
        self.w3 = NIULinear(d_model, d_ff, device = device, dtype = dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.device == self.w1.weight.device
        assert x.device == self.w2.weight.device
        assert x.device == self.w3.weight.device
        a = self.w1(x)
        b = NIUSiLU()(a)
        
        c = self.w3(x)
        
        bc = b * c
        return self.w2(bc)
```
本质上我们需要求w1,w2,w3三个张量的梯度，所以是他们三个的激活值。
对于w1，由a = self.w1(x)是和x相乘，所以激活值和x张量大小一致，是：B * L * H；
对于w2，由 self.w2(bc) 是和bc，而bc大小是：B * L * MH
对于w3，由c = self.w3(x) ，显而易见，激活值的大小是 ：B * L * H；

对于B = 10， L = 200， H = 512， MH = 1344的设定来说， 精度是fp32，4字节
ffn总的激活大小是： 10 * 200 * （2 * 512 + 1344）* 4 / 1024/ 1024 = 18.07MB
# 十一、Now look closely at the “Active Memory Timeline” from pytorch.org/memory_viz of a memory snapshot of the 2.7B model doing a forward pass. When you reduce the “Detail” level, the tool hides the smallest allocations to the corresponding level (e.g., putting “Detail” at 10% only shows 8 the 10% largest allocations). What is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?

![显存分配情况](./statics/memory_profile_case.png)
上面分配显存最大的是在transformerLM最后一个 output = self.lm_head(x)模块，这个模块将隐藏层维度映射为字典大小
一共占了57.2MiB

善后可以看到有四个小尖尖，这是前向传播经过四层。
并且每一个尖尖对应三个条带，然后我仔细地看了一下，按照时间先后顺序分表表示多的是exp操作分配地张量13.7MiB，exp操作分配地张量13.7MiB，然后张量除法分配地显存13.7MiB。
有点感慨，我自己写的简单一行代码居然需要申请三次显存。


# 十二、Report the timings (or out-of-memory errors) you get for these configurations. At what size do you get out-of-memory errors? Do the accounting for the memory usage of attention in one of the smallest configurations you find that runs out of memory (you can use the equations for memory usage of Transformers from Assignment 1). How does the memory saved for backward change with the sequence length? What would you do to eliminate this memory cost
当序列长度是16384在我们的4060Ti显卡上面显存爆炸了；
第二个问题我是没有太读懂，反正就是自己计算的显存占用肯定小于实际显存的占用，下面我可以针对一个操作说明一下：
```python
 attention_score_matrix = dot_product/torch.sqrt(torch.tensor(d_k, device=dot_product.device))
```
这里首先存储dot_product 需要 4 * B * L * L B的显存大小，torch.sqrt(torch.tensor(d_k, device=dot_product.device)) 这一段也有可能创建了和dot_product
一样大小的显存，最终为了保存除法结果，我们还得创建attention_score_matrix这个结果张量 4 * B * L * L B 
所以一共是 3 *  4 * B * L * L B


![attention 的前向和后向传播情况](./statics/profile_attention_memory.png)
```python
import torch
import torch.nn as nn
from einops import rearrange, einsum

from jaxtyping import Float, Bool
from torch import Tensor
import torch.cuda.nvtx as nvtx

from cs336_basics.modules.softmax import NIUsoftmax

class NIUscaled_dot_product_attention(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NIUscaled_dot_product_attention, self).__init__()
        pass
    def forward(self,
                Q: Float[Tensor, "... queries d_k"],
                K: Float[Tensor, "... keys d_k"],
                V: Float[Tensor, "... keys d_v"],
                mask: Bool[Tensor, " ... queries keys"] | None = None)\
                -> Float[Tensor, " ... queries d_v"]:

                mask = mask.to(Q.device)
                
                d_k = Q.shape[-1]
                with nvtx.range("Q @ K^T matrix calculation"):
                    dot_product = einsum(Q, K, '... queries d_k, ... keys d_k -> ... queries keys')
                attention_score_matrix = dot_product/torch.sqrt(torch.tensor(d_k, device=dot_product.device))
                # if mask mask some scores
                if mask is not None:
                    masked_attention_score_matrix = attention_score_matrix.masked_fill_(~mask.bool(), -float('inf'))
                else:
                    masked_attention_score_matrix = attention_score_matrix
                
                # with nvtx.range("softmax process"):
                #     softmax_attention_score_matrix = NIUsoftmax()(masked_attention_score_matrix, dim=-1)
                
                with nvtx.range("softmax_attention_score_matrix @ V sum calculation"):
                    result = einsum(masked_attention_score_matrix, V, '... queries keys, ... keys d_v -> ... queries d_v')
                return result
```
上面图片中第二个灰色开始的阶段就是backward开始的阶段。后面三个像闪电一样的东西其实是创建的三个梯度矩阵，从前往后分别是：
- torch::autograd::generated::BmmBackward0::apply
- torch::autograd::generated::MaskedFillBackward0::apply
- torch::autograd::generated::DivBackward0::apply

后向传播主要的显存构成有：模型的参数、模型的梯度、优化器的额外参数（比如adamW的一阶动量和二阶动量）、GPU计算图一级反向传播必要的激活值
在这个attention的反向计算过程也和上面图片展示的是一模一样的，第一阶段：我们的梯度到达这个attention的时候，首先计算的是V的梯度，
V的梯度是：
$$
\begin{aligned}
\frac{\partial L}{\partial V} 
&= \frac{\partial L}{\partial \text{result}} \cdot \frac{\partial \text{result}}{\partial V} \\
&= \left( \text{masked\_attention\_score\_matrix} \right)^\top \frac{\partial L}{\partial \text{result}}
\end{aligned}
这时候需要的显存大头是masked\_attention\_score\_matrix和masked\_attention\_score\_matrix的梯度（因为要进一步反向传播)
$$
第二阶段：我们需要经过mask的反向传播，这个时候释放masked_attention_score_matrix的显存，然后重新申请attention_score_matrix的显存，他们是一样大的，mask的反向传播就是没有被masked的保持原样，masked的部分梯度变为0

第三阶段：我们求取的是Q和K的梯度，相关的公式是：
$$
\begin{aligned}
\frac{\partial L}{\partial Q} 
&= \frac{\partial L}{\partial \text{attention\_score\_matrix}} \cdot \frac{\partial \text{attention\_score\_matrix}}{\partial Q} \\
&=  \frac{\partial L}{\partial \text{attention\_score\_matrix}} \left( \text{K} \right)
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial L}{\partial K} 
&= \frac{\partial L}{\partial \text{attention\_score\_matrix}} \cdot \frac{\partial \text{attention\_score\_matrix}}{\partial K} \\
&=  \frac{\partial L}{\partial \text{attention\_score\_matrix}}^\top \left( \text{Q} \right)
\end{aligned}
$$
在这一个阶段显存大头是$\frac{\partial L}{\partial \text{attention\_score\_matrix}}$

补充：计算完基础的$QK^\top$之后要进行除法，这个也是反向传播计算梯度，只是梯度是常数。





