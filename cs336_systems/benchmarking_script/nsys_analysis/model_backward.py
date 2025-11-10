import torch
from cs336_basics.modules.TransformerLM import NiuTransformerLM
import torch.cuda as tc
import torch.cuda.nvtx as nvtx
from cs336_basics.optimizer.optimizer import NIUAdam
def model_backward():
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
    logits = model(dummy_data)
    loss = torch.nn.functional.cross_entropy(logits, logits)
    with nvtx.range("model_backward_pass"):
        loss.backward()
        
        torch.cuda.synchronize()
    return logits, loss

def main():
    model_backward()






if __name__ == "__main__":
    main()