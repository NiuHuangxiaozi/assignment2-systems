import  torch
import argparse
import torch.nn as nn
import timeit
from cs336_basics.modules.scaled_dot_product_attention import NIUscaled_dot_product_attention
from cs336_basics.modules.TransformerLM import NiuTransformerLM
torch.set_float32_matmul_precision('high')

def main(args):
    print("Starting torch compile")

    if args.model == "attention":
        Q = torch.randn(1, 1024, 512, device="cuda",requires_grad=True)
        K = torch.randn(1, 1024, 512, device="cuda",requires_grad=True)
        V = torch.randn(1, 1024, 512, device="cuda",requires_grad=True)
        mask = torch.tril(torch.ones(1024, 1024, device="cuda")).bool()
        compiled_model = torch.compile(NIUscaled_dot_product_attention().to("cuda"))
        model = NIUscaled_dot_product_attention().to("cuda")
        
        def forward_original(model, Q, K, V, mask):
            result = model(Q, K, V, mask)
            torch.cuda.synchronize()
            return result
        
        def forward_compiled(compiled_model, Q, K, V, mask):
            result = compiled_model(Q, K, V, mask)
            torch.cuda.synchronize()
            return result
        # compile warm up
        for _ in range(100):
            _ = compiled_model(Q, K, V, mask)
        torch.cuda.synchronize()
        
        
        # begin to estimate time  
        original_time_mean = timeit.timeit(lambda: forward_original(model, Q, K, V, mask), number=100)
        print(f"Original Time taken: {original_time_mean}")
        
        compiled_time_mean = timeit.timeit(lambda: forward_compiled(compiled_model, Q, K, V, mask), number=100)
        print(f"Compiled Time taken: {compiled_time_mean}")
        
        print(f"Speedup: { original_time_mean / compiled_time_mean}")
    elif args.model == "transformerLM":
        model = NiuTransformerLM(10000, 128, 512, 4, 16, 1344, 10000, device="cuda")
        compiled_model = torch.compile(model)
        dummy_data = torch.randint(0, 10000, (128, 128)).to("cuda")
        
        def forward_original(model, dummy_data):
            result = model(dummy_data)
            torch.cuda.synchronize()
            return result
        
        def forward_compiled(compiled_model, dummy_data):
            result = compiled_model(dummy_data)
            torch.cuda.synchronize()
            return result
        
        for _ in range(15):
            result = compiled_model(dummy_data)
        torch.cuda.synchronize()
        
        original_time_mean = timeit.timeit(lambda: forward_original(model, dummy_data), number=15)
        print(f"Original Time taken: {original_time_mean}")
        
        compiled_time_mean = timeit.timeit(lambda: forward_compiled(compiled_model, dummy_data), number=15)
        print(f"Compiled Time taken: {compiled_time_mean}")
        
        print(f"Speedup: { original_time_mean / compiled_time_mean}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("use torch compile to optimize the model")
    parser.add_argument("--torch_compile", action="store_true", default=False)
    parser.add_argument("--model", type=str, default="attention", choices=["attention", "transformerLM"])
    args = parser.parse_args()
    main(args)