import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Any



class NIUNativeDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super(NIUNativeDDP, self).__init__()
        self.module = module
        self.work_list = []

        # initialize all parameters to be the same
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.grad_change)

    

    def grad_change(self, param: torch.nn.Parameter):
            param.grad.data /= dist.get_world_size()
            new_work = dist.all_reduce(param.grad.data, op = dist.ReduceOp.SUM, async_op = True)
            self.work_list.append(new_work)
    
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)
    
    def wait_for_all_gradients(self):
        for work in self.work_list:
            work.wait()
        self.work_list.clear()
