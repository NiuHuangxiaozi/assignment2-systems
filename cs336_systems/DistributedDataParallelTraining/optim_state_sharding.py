import torch
from typing import Type, Any
import torch.distributed as dist
from typing import List, Any, Optional, Callable

class NIUOptimStateSharding(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):

        # watch out params is a generator, we need to convert it to a list
        self.all_params = list(params)
        super().__init__(self.all_params, {})

        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        self.cur_params = [ param for i, param in enumerate(self.all_params) if i % self.world_size == self.rank ]
        self.optimizer = optimizer_cls(self.cur_params, **kwargs)
        
        self.handles = []
    #
    def step(self, closure: Optional[Callable] = None):
        '''
            update the parameters in the sharded_params
            synchronize with the other ranks' parameters.
            
        '''
        # update the parameters in the sharded_params
        self.optimizer.step()         # synchronize with the other ranks' parameters.
        self.synchronize_params()
        self.wait_for_all_params()

    def synchronize_params(self):

        for i, param in enumerate(self.all_params):
            rank = i % self.world_size
            self.handles.append(dist.broadcast(param.data, src=rank, async_op=True))

    def wait_for_all_params(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
