import torch
from cs336_basics.modules.TransformerLM import NiuTransformerLM
import torch.cuda as tc
import torch.cuda.nvtx as nvtx
from cs336_basics.optimizer.optimizer import NIUAdam
from cs336_basics.loss.loss import NIUCrossEntropyLoss
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
    niu_optimizer = NIUAdam(model.parameters(), lr=0.001)
    dummy_data = torch.randint(0, 10000, (10, 256)).to(device)
    labels = torch.randint(0, 10000, (10,)).to(device)
    for _ in range(100):
        niu_optimizer.zero_grad()
        logits = model(dummy_data)
        loss = NIUCrossEntropyLoss()(logits, labels)
        loss.backward()
        niu_optimizer.step()
        torch.cuda.synchronize()
    return logits, loss

def main():
    model_backward()






if __name__ == "__main__":
    main()