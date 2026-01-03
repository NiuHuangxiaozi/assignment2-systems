import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Any




# https://github.com/heng380/cs336_assignment2/blob/main/cs336_systems/paralism/individual_bucketed_ddp.py
class Bucket:
    def __init__(self, num_params: int):
        self.num_params = num_params
        self.params = []

    # return None if the bucket is not full, otherwise return the allreduce result
    def add_param(self, param: torch.nn.Parameter):
        self.params.append(param)
        if len(self.params) >= self.num_params:
            output = self.allreduce()
            self.params = []
            return output
        return None

    # return the allreduce result
    def allreduce(self):
        all_reduce_tensor = torch._utils._flatten_dense_tensors(tensors = [p.grad for p in self.params])
        all_reduce_tensor /= dist.get_world_size()
        new_work = dist.all_reduce(all_reduce_tensor, op = dist.ReduceOp.SUM, async_op = True)
        return new_work, self.params, all_reduce_tensor

class NIUBucketDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_Mb: float):
        super(NIUBucketDDP, self).__init__()
        self.module = module
        self.bucket_results = []
        self.param_to_bucket = {}

        self.bucket_size_bytes = bucket_size_Mb * 1024 * 1024 

        self.buckets = []


        # initialize all parameters to be the same
        # define which parameters to bucket
        # param.data.nbytes
        curr_bytes = 0
        curr_bucket_params = [] 
        for param in reversed(list(self.module.parameters())):
            dist.broadcast(param.data, src=0)
            if not param.requires_grad:
                continue
            curr_bytes += param.data.nbytes
            curr_bucket_params.append(param)
            
            if curr_bytes >= self.bucket_size_bytes:
                new_bucket = Bucket(len(curr_bucket_params))

                # connect the parameters to the bucket
                for param in curr_bucket_params:
                    self.param_to_bucket[param] = new_bucket

                # add the bucket to the list
                self.buckets.append(new_bucket)
                curr_bucket_params = []
                curr_bytes = 0

            param.register_post_accumulate_grad_hook(self.buffer_grad_hook)

        if curr_bucket_params:
            new_bucket = Bucket(len(curr_bucket_params))
            for param in curr_bucket_params:
                self.param_to_bucket[param] = new_bucket
            self.buckets.append(new_bucket)


    def buffer_grad_hook(self, param: torch.nn.Parameter):
        bucket = self.param_to_bucket[param]
        output = bucket.add_param(param)
        if output is not None:    
            self.bucket_results.append(output)
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        for results, params, all_reduce_tensor in self.bucket_results:
            results.wait()

            unflattened_grad = torch._utils._unflatten_dense_tensors(all_reduce_tensor, params)
            for param, new_grad in zip(params, unflattened_grad):
                param.grad.data = new_grad

        self.bucket_results.clear()

