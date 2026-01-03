import torch
import os
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import Callable
from statistics import mean
import argparse


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    dist.init_process_group(backend = backend, rank = rank, world_size = world_size)

def allreduce_demo(rank, world_size, data_size, num_warmups, backend):
    
    setup(rank, world_size, backend=backend)
    
    data_number = data_size * 1024 * 1024 // 4 # 4 bytes per int
    
    # device
    if backend == "Gloo":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{rank}")

    if backend == "NCCL":
        # warm up
        tmpdata = torch.randint(0, 10, (data_number,), device=device)
        for _ in range(num_warmups):
            dist.all_reduce(tmpdata, op = dist.ReduceOp.SUM)
            torch.cuda.synchronize()

    data = torch.randint(0, 10, (data_number,), device=device)
    # print(f"Rank {rank} has data: {data.tolist()}")
    start_time = time.time()
    dist.all_reduce(data, op = dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    end_time = time.time()
    allreduce_time = torch.tensor([end_time - start_time])

    # rank 0 collect all the times
    all_times = [torch.zeros_like(allreduce_time) for _ in range(world_size)]
    dist.all_gather(all_times, allreduce_time)
    if rank == 0:
        # print(f"Allreduce result: {data.tolist()}")
        all_times = torch.stack(all_times)
        print(f"Allreduce times: {all_times.tolist()}")
        mean_time = all_times.mean()
        std_time = all_times.std()
        print(f"Mean allreduce time: {mean_time.item()}")
        print(f"Std allreduce time: {std_time.item()}")
    dist.destroy_process_group()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark allreduce performance")
    parser.add_argument("--world_size", type=int, default=2, help="number of processes")
    parser.add_argument("--data_size", type=int, default=4, help="size of data in MB")
    parser.add_argument("--num_warmups", type=int, default=5, help="number of warmups")
    parser.add_argument("--backend", type=str, default="Gloo", help="backend to use")
    args = parser.parse_args()
    world_size = args.world_size
    data_size = args.data_size
    num_warmups = args.num_warmups
    backend = args.backend
    mp.spawn(fn = allreduce_demo, args = (world_size, data_size, num_warmups, backend), nprocs = world_size, join = True)