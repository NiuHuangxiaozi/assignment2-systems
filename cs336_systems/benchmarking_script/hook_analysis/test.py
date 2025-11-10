import time
import torch
import torch.cuda as tc
import numpy as np
from cs336_basics.modules.TransformerLM import NiuTransformerLM
def add_timing_hooks(model):
    times = {}

    def hook_wrapper(name):
        def hook(module, input):
            start = time.time()
            torch.cuda.synchronize()
            module._start_time = start
        return hook

    def hook_wrapper_post(name):
        def hook(module, input, output):    
            torch.cuda.synchronize()
            end = time.time()
            elapsed = end - module._start_time
            times[name] = times.get(name, []) + [elapsed]
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 只hook叶子模块
            module.register_forward_pre_hook(hook_wrapper(name))
            module.register_forward_hook(hook_wrapper_post(name))

    return times



def test():
    device = "cuda:0" if tc.is_available() else "cpu"
    model = NiuTransformerLM(10000,
                             256,
                             512,
                             4,
                             16,
                             1344,
                             10000,
                             device=device)
    times = add_timing_hooks(model)
    dummy_data = torch.randint(0, 10000, (2, 256)).to(device)
    _ = model(dummy_data)  # 执行一次前向传播
    print("=== Forward time breakdown ===")
    for name, t in times.items():
        print(f"{name}: mean={np.mean(t):.6f}s, std={np.std(t):.6f}s")

    print("=== Forward time Analysis ===")
    # 在你已有 times dict 之后添加：
    total = sum(np.mean(t) for t in times.values())

    # 构造一个列表，每项 (name, mean_time)
    stat = [(name, np.mean(t)) for name, t in times.items()]
    # 按 mean_time 从大到小排序
    stat.sort(key=lambda x: x[1], reverse=True)

    print("=== Forward time breakdown (sorted) ===")
    print(f"{'Module':60s} {'Mean(s)':>10s} {'% of total':>12s}")
    for name, mean_time in stat:
        percent = mean_time / total * 100 if total > 0 else 0.0
        print(f"{name:60s} {mean_time:10.6f} {percent:12.2f}%")
    print(f"{'TOTAL':60s} {total:10.6f} {100.00:12.2f}%")
    
if __name__ == "__main__":
    test()