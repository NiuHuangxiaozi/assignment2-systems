
import argparse
import torch
import torch.cuda as tc
import timeit
import numpy as np
from cs336_basics.modules.TransformerLM import NiuTransformerLM
from cs336_basics.train.train_utils import load_model_config




def model_forward_pass_timeit(model, dummy_data):
    
    logits = model(dummy_data)
    torch.cuda.synchronize()
    return logits


def model_backward_pass_timeit(model, loss):
    loss.backward(retain_graph=True)
    torch.cuda.synchronize()

def calculate_forward_pass_time(model, dummy_data, run_number):
    times = []
    for _ in range(run_number):
        t = timeit.timeit(lambda: model_forward_pass_timeit(model, dummy_data), number=1)
        times.append(t)

    # 计算统计量
    mean_time = sum(times) / run_number
    # 如果你用 numpy：
    std_time = np.std(times, ddof=0)  # ddof=0 表示总体标准差
    # 或者如果想样本标准差：
    std_sample = np.std(times, ddof=1)
    print(f"="*10)
    print(f"Runs: {run_number}")
    print(f"Times: {times}")
    print(f"Mean forward pass time: {mean_time:.6f} seconds")
    print(f"Std deviation (population): {std_time:.6f} seconds")
    print(f"Std deviation (sample): {std_sample:.6f} seconds")
    print(f"="*10)
    

def calculate_backward_pass_time(model, dummy_data, run_number):
    
    logits = model(dummy_data)
    loss = torch.nn.functional.cross_entropy(logits, logits)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(run_number):
        t = timeit.timeit(lambda: model_backward_pass_timeit(model, loss), number=1)
        times.append(t)
    # 计算统计量
    mean_time = sum(times) / run_number
    # 如果你用 numpy：
    std_time = np.std(times, ddof=0)  # ddof=0 表示总体标准差
    # 或者如果想样本标准差：
    std_sample = np.std(times, ddof=1)
    print(f"="*10)
    print(f"Runs: {run_number}")
    print(f"Times: {times}")
    print(f"Mean backward pass time: {mean_time:.6f} seconds")
    print(f"Std deviation (population): {std_time:.6f} seconds")
    print(f"Std deviation (sample): {std_sample:.6f} seconds")
    print(f"="*10)
def main(args):

    # 选择设备
    device = "cuda" if tc.is_available() else "cpu"
    
    # 创建模型
    model_cfg = load_model_config(args.model_cfg_path)
    model = NiuTransformerLM(model_cfg.vocab_size,
                             model_cfg.context_length,
                             model_cfg.d_model,
                             model_cfg.num_layers,
                             model_cfg.num_heads,
                             model_cfg.d_ff,
                             model_cfg.rope_theta,
                             device=device)

    
    # 创建dummpy数据
    dummy_data = torch.randint(0, model_cfg.vocab_size, (128, model_cfg.context_length)).to(device)
    
    
    
    def warm_up(model, dummy_data):
        logits = model(dummy_data)
        loss = torch.nn.functional.cross_entropy(logits, logits)
        loss.backward()
        return logits, loss
    warm_up_time_list = []
    warm_up_steps = args.warm_up_steps
    for _ in range(warm_up_steps):
        t = timeit.timeit(lambda: warm_up(model, dummy_data), number=1)
        warm_up_time_list.append(t)
    print(f"Warm up time: {warm_up_time_list} seconds")
    
    
    # 测试forward pass时间
    print("Testing forward pass time...")
    calculate_forward_pass_time(model, dummy_data, 10)
    
    print("Testing backward pass time...")
    calculate_backward_pass_time(model, dummy_data, 10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("test time cost of forward pass and backward pass of transformerLM")
    parser.add_argument("--model_cfg_path", type=str, default="/home/niu/code/cs336/assignment2-systems/cs336_systems/benchmarking_script/timeit_strategy/model_configs.yaml")
    parser.add_argument("--warm_up_steps", type=int, default=0)
    parser.add_argument("--run_number", type=int, default=10)
    args = parser.parse_args()
    main(args)