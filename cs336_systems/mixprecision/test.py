
import torch
import argparse
from cs336_basics.modules.TransformerLM import NiuTransformerLM
import torch.cuda as tc
from cs336_basics.loss.loss import NIUCrossEntropyLoss
from cs336_basics.optimizer.optimizer import NIUAdam
import timeit
def model_forward(d_model: int, 
                  d_ff: int,
                  num_layers: int,
                  num_heads: int):
    device = "cuda:0" if tc.is_available() else "cpu"
    model = NiuTransformerLM(10000,
                             256,
                             d_model,
                             num_layers,
                             num_heads,
                             d_ff,
                             10000,
                             device=device)
    dummy_data = torch.randint(0, 10000, (2, 10)).to(device)
    dummy_data_labels = torch.randint(0, 1000, (2,)).to(device)
    optimizer = NIUAdam(model.parameters(), lr=0.001)
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(dummy_data)
        loss = NIUCrossEntropyLoss()(logits[:,-1,:], dummy_data_labels)
        loss.backward()
        optimizer.step()
    return loss   
def main():
    parser = argparse.ArgumentParser("test mixprecision of transformerLM")
    parser.add_argument("--autocast", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, default="small")
    
    args = parser.parse_args()
    if args.autocast:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if args.model_size == "small":
                small_model_time = timeit.timeit(lambda: model_forward(d_model=768, d_ff=3072, num_layers=2, num_heads=12), number=3)
                print(f"Autocast Small Model Time: {small_model_time} seconds")
            elif args.model_size == "medium":
                medium_model_time = timeit.timeit(lambda: model_forward(d_model=1024, d_ff=4096, num_layers=3, num_heads=16), number=3)
                print(f"Autocast Medium Model Time: {medium_model_time} seconds")
            elif args.model_size == "large":
                large_model_time = timeit.timeit(lambda: model_forward(d_model=1280, d_ff=5120, num_layers=4, num_heads=20), number=3)
                print(f"Autocast Large Model Time: {large_model_time} seconds")
    else:
        if args.model_size == "small":
            small_model_time = timeit.timeit(lambda: model_forward(d_model=768, d_ff=3072, num_layers=2, num_heads=12), number=3)
            print(f"No Autocast Small Model Time: {small_model_time} seconds")
        elif args.model_size == "medium":
            medium_model_time = timeit.timeit(lambda: model_forward(d_model=1024, d_ff=4096, num_layers=3, num_heads=16), number=3)
            print(f"No Autocast Medium Model Time: {medium_model_time} seconds")
        elif args.model_size == "large":
            large_model_time = timeit.timeit(lambda: model_forward(d_model=1280, d_ff=5120, num_layers=4, num_heads=20), number=3)
            print(f"No Autocast Large Model Time: {large_model_time} seconds")

    print(f"over")

if __name__ == "__main__":
    main()