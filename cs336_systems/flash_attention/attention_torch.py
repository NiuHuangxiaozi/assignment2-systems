import torch
import torch.nn as nn

# 普通的self-attention代码实现
class NIUAttention(nn.Module):
    def __init__(self,):
        super(NIUAttention, self).__init__()
        self.name = "NIUAttention"

    def __str__(self):
        return self.name

    def forward(self, Q, K, V, is_causal=False):
        
        d = Q.shape[-1]
        scale = 1 / (d ** 0.5)
        S=Q @ K.transpose(-2, -1) * scale
        if is_causal:
            S = S.masked_fill(~torch.tril(torch.ones_like(S)).bool(), -float('inf'))
        P = torch.softmax(S, dim=-1)
        o = P @ V
        return o