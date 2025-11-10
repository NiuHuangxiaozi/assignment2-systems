
import torch
import argparse
from cs336_basics.modules.TransformerLM import NiuTransformerLM
import torch.cuda as tc
import torch.cuda.nvtx as nvtx


def model_forward(args):
    device = "cuda:0" if tc.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.init()
        # 触发 context
        _ = torch.empty((1,), device=device)
        torch.cuda.synchronize()
    model = NiuTransformerLM(10000,
                             256,
                             512,
                             4,
                             16,
                             1344,
                             10000,
                             device=device)
    dummy_data = torch.randint(0, 10000, (10, 256)).to(device)
    for _ in range(100):
        logits = model(dummy_data)
    torch.cuda.synchronize()
    nvtx.range_pop();
    return logits   
def main():
    parser = argparse.ArgumentParser("test time cost of forward pass and backward pass of transformerLM")
    parser.add_argument("--model_cfg_path", type=str, default="/assignment2-systems/cs336_systems/benchmarking_script/timeit_strategy/model_forward.py")
    args = parser.parse_args()
    model_forward(args)



if __name__ == "__main__":
    main()