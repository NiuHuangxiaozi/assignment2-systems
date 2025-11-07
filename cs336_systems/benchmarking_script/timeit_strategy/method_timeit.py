
import argparse
import torch
import torch.cuda as tc
import timeit
from cs336_basics.modules.TransformerLM import NiuTransformerLM
from cs336_basics.train.train_utils import load_model_config


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
    dummy_data = torch.randint(0, model_cfg.vocab_size, (1, model_cfg.context_length), device=device)
    
    
    warm_up_steps = 10
    for _ in range(warm_up_steps):
        model(dummy_data)
    
    # 测试forward pass时间
    forward_time = timeit.timeit(lambda: model(dummy_data), number=10) / 10
    print(f"Forward pass time: {forward_time} seconds")
    
    # 测试backward pass时间
    backward_time = timeit.timeit(lambda: model(dummy_data).backward(), number=10) / 10
    print(f"Backward pass time: {backward_time} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("test time cost of forward pass and backward pass of transformerLM")
    parser.add_argument("--model_cfg_path", type=str, default="timeit_strategy/model_configs.yaml")
    args = parser.parse_args()
    main(args)