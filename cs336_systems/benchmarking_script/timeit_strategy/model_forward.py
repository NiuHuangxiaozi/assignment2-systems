
import torch
import argparse
from cs336_basics.modules.TransformerLM import NiuTransformerLM
from cs336_basics.train.train_utils import load_model_config
import torch.cuda as tc
import torch.cuda.nvtx as nvtx


def model_forward(args):
    nvtx.range_push("total forward_pass")
    device = "cuda:0" if tc.is_available() else "cpu"
    model_cfg = load_model_config(args.model_cfg_path)
    model = NiuTransformerLM(model_cfg.vocab_size,
                             model_cfg.context_length,
                             model_cfg.d_model,
                             model_cfg.num_layers,
                             model_cfg.num_heads,
                             model_cfg.d_ff,
                             model_cfg.rope_theta,
                             device=device)
    dummy_data = torch.randint(0, model_cfg.vocab_size, (2, model_cfg.context_length)).to(device)
    logits = model(dummy_data)
    torch.cuda.synchronize()
    return logits   
def main():
    parser = argparse.ArgumentParser("test time cost of forward pass and backward pass of transformerLM")
    parser.add_argument("--model_cfg_path", type=str, default="/home/niu/code/cs336/assignment2-systems/cs336_systems/benchmarking_script/timeit_strategy/model_configs.yaml")
    args = parser.parse_args()
    model_forward(args)



if __name__ == "__main__":
    main()