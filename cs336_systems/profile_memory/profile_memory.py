import torch
import argparse
from cs336_basics.modules.TransformerLM import NiuTransformerLM
from cs336_basics.optimizer.optimizer import NIUAdam
from cs336_basics.loss.loss import NIUCrossEntropyLoss


def profile_memory(args):
    if args.mode == "forward":
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        model = NiuTransformerLM(10000,
                                256,
                                512,
                                4,
                                16,
                                1344,
                                10000,
                                device="cuda")
        dummy_data = torch.randint(0, 10000, (10, args.context_length)).to("cuda")
        for _ in range(3):
            model(dummy_data)
        torch.cuda.memory._dump_snapshot(f"forward_memory_snapshot_{args.context_length}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        print("Forward pass memory profiling completed.")
    else:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        model = NiuTransformerLM(10000,
                                256,
                                512,
                                4,
                                16,
                                1344,
                                10000,
                                device="cuda")
        niu_optimizer = NIUAdam(model.parameters(), lr=0.001)
        dummy_data = torch.randint(0, 10000, (10, args.context_length)).to("cuda")
        labels = torch.randint(0, 10000, (10,)).to("cuda")
        for _ in range(3):
            niu_optimizer.zero_grad()
            logits = model(dummy_data)
            predict = logits[:,-1,:]
            loss = NIUCrossEntropyLoss()(predict, labels)
            loss.backward()
            niu_optimizer.step()
        torch.cuda.memory._dump_snapshot(f"forward_and_backward_memory_snapshot_{args.context_length}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        print("Forward and Backward pass memory profiling completed.")
        
        
def main():
    print("Starting memory profiling...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="forward")
    parser.add_argument("--context_length", type=int, default=256)
    args = parser.parse_args()  
    profile_memory(args)

if __name__ == "__main__":
    main()