import argparse
import timeit
import torch.nn as nn
import torch
from cs336_basics.modules.scaled_dot_product_attention import NIUscaled_dot_product_attention
from cs336_basics.optimizer.optimizer import NIUAdam
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='attention_timeit.log')

def profile_attention(args):
    if args.exp_name == "attention_timeit": 
        # 初始化输入qkv和模型
        model = NIUscaled_dot_product_attention().to("cuda")
        if args.model_mode == "forward":
            def attention_timeit(model, Q, K, V, mask):
                attention_result = model(Q, K, V, mask)
                return attention_result
            
            for d_model in [16,32,64,128]:
                for input_length in [256, 1024, 4096, 8192, 16384]:
                    Q = torch.randn(args.batchsize, input_length, d_model, device="cuda",requires_grad=True)
                    K = torch.randn(args.batchsize, input_length, d_model, device="cuda",requires_grad=True)
                    V = torch.randn(args.batchsize, input_length, d_model, device="cuda",requires_grad=True)
                    mask = torch.tril(torch.ones(input_length, input_length, device = "cuda")).bool()
                    logging.info(f"[Forward]: Profiling attention timeit for d_model {d_model} and input length {input_length}...")
                    time_mean = timeit.timeit(lambda: attention_timeit(model, Q, K, V, mask), number=args.timeit_num)
                    logging.info(f"Mean time: {time_mean}")
        elif args.model_mode == "backward":
            def attention_timeit(model, Q, K, V, mask, optimizer):
                attention_result = model(Q, K, V, mask)
                loss = attention_result.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                return loss.item()

            for d_model in [16,32,64,128]:
                for input_length in [256, 1024, 4096, 8192, 16384]:
                    Q = torch.randn(args.batchsize, input_length, d_model,device="cuda",requires_grad=True)
                    K = torch.randn(args.batchsize, input_length, d_model,device="cuda",requires_grad=True)
                    V = torch.randn(args.batchsize, input_length, d_model,device="cuda",requires_grad=True)
                    mask = torch.tril(torch.ones(input_length, input_length, device="cuda")).bool()
                    optimizer = NIUAdam(params=[Q, K, V], lr=0.001)
                    logging.info(f"[Backward]: Profiling attention timeit for d_model {d_model} and input length {input_length}...")
                    time_mean = timeit.timeit(lambda: attention_timeit(model, Q, K, V, mask, optimizer), number=args.timeit_num)
                    logging.info(f"Mean time: {time_mean}")
        else:
            raise ValueError(f"Invalid model mode: {args.model_mode}")
    elif args.exp_name == "attention_memory_summary":
        if args.model_mode == "forward":
            for d_model in [16,32,64,128]:
                for input_length in [256, 1024, 4096, 8192, 16384]:
                    torch.cuda.memory._record_memory_history(max_entries=1000000)
                    model = NIUscaled_dot_product_attention().to("cuda")
                    Q = torch.randn(args.batchsize, input_length, d_model, device="cuda",requires_grad=True)
                    K = torch.randn(args.batchsize, input_length, d_model, device="cuda",requires_grad=True)
                    V = torch.randn(args.batchsize, input_length, d_model, device="cuda",requires_grad=True)
                    mask = torch.tril(torch.ones(input_length, input_length, device="cuda")).bool()
                    model(Q, K, V, mask)
                    torch.cuda.memory._dump_snapshot(f"forward_memory_snapshot_{args.batchsize}_{d_model}_{input_length}_{'torch_compile' if args.use_torch_compile else 'no_torch_compile'}.pickle")
                    torch.cuda.memory._record_memory_history(enabled=None)  
            print("Forward pass memory profiling completed.")
        elif args.model_mode == "backward":
            for d_model in [16,32,64,128]:
                for input_length in [256, 1024, 4096, 8192, 16384]:
                    torch.cuda.memory._record_memory_history(max_entries=1000000)
                    model = NIUscaled_dot_product_attention().to("cuda")
                    Q = torch.randn(args.batchsize, input_length, d_model, device="cuda",requires_grad=True)
                    K = torch.randn(args.batchsize, input_length, d_model, device="cuda",requires_grad=True)
                    V = torch.randn(args.batchsize, input_length, d_model, device="cuda",requires_grad=True)
                    niu_optimizer = NIUAdam(params=[Q, K, V], lr=0.001)
                    mask = torch.tril(torch.ones(input_length, input_length, device="cuda")).bool()
                    attention_result = model(Q, K, V, mask)
                    loss = attention_result.sum()
                    niu_optimizer.zero_grad()
                    loss.backward()
                    niu_optimizer.step()
                    torch.cuda.memory._dump_snapshot(f"backward_memory_snapshot_{args.batchsize}_{d_model}_{input_length}_{'torch_compile' if args.use_torch_compile else 'no_torch_compile'}.pickle")
                    torch.cuda.memory._record_memory_history(enabled=None)
                    print("Backward pass memory profiling completed.")
        else:
            raise ValueError(f"Invalid model mode: {args.model_mode}")
    else:
        raise ValueError(f"Invalid experiment name: {args.exp_name}")
    print("Attention profiling completed.")

def main():
    # 这个py文件是使用timeit来探究attention算子对于输入序列长度的影响
    # 同时我们也将探究torch.compile对于模型的优化
    print("Starting attention profiling using timeit and torch.memory_summary")
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="attention_timeit", choices=["attention_timeit", "attention_memory_summary"])
    parser.add_argument("--model_mode", type=str, default="forward", choices=["forward", "backward"])
    parser.add_argument("--timeit_num", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--head_num", type=int, default=1)
    parser.add_argument("--use_torch_compile", action="store_true", default=False)
    args = parser.parse_args()  
    profile_attention(args)


if __name__ == "__main__":
    main()



