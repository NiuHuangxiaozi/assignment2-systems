import torch



torch.random.manual_seed(0)
batch_size = 4
n_queries = 128
n_keys = 128
D = 64
q = torch.randn(batch_size, n_queries, D, device="cuda",requires_grad=True)
k = torch.randn(batch_size, n_keys, D, device="cuda",requires_grad=True)
v = torch.randn(batch_size, n_keys, D, device="cuda",requires_grad=True)
do = torch.randn(batch_size, n_queries, D, device="cuda")
torch.save(q, "./temp_qkvdo/q.pt")
torch.save(k, "./temp_qkvdo/k.pt")
torch.save(v, "./temp_qkvdo/v.pt")
torch.save(do, "./temp_qkvdo/do.pt")