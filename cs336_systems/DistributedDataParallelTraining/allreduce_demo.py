import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp



def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    dist.init_process_group(backend = "gloo", rank = rank, world_size = world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (4,))
    print(f"Rank {rank} has data: {data}")
    dist.all_reduce(data, op = dist.ReduceOp.SUM)
    print(f"Rank {rank} has reduced data: {data}")
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 3
    mp.spawn(fn = distributed_demo, args = (world_size,), nprocs = world_size, join = True)
