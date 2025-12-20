import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import einsum
from typing import Tuple




# 使用torch的语言实现flashattention
class NIUFlashAttentionV2_torch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 准备超参数
        B_q = 16
        B_k = 16
        hidden_embedding_type = Q.dtype

        Q = Q.to(torch.float32)
        K = K.to(torch.float32)


        batch, seq_len, d = Q.shape
        query_shape = Q.shape #  (batch, seq_len, d)
        device = Q.device
    
        splited_Q: Tuple[torch.Tensor, ...] = torch.split(Q, B_q, dim=-2)
        splited_K: Tuple[torch.Tensor, ...] = torch.split(K, B_k, dim=-2)
        splited_V: Tuple[torch.Tensor, ...] = torch.split(V, B_k, dim=-2)
        
        # 计算T_q和T_k，准备循环
        T_q = len(splited_Q)
        T_k = len(splited_K)

        O_list = []
        L_list = []
        # 外层先加载kv，内层加载query
        for i in range(0, T_q):
            # 加载query
            q_i = splited_Q[i]
            # assert q_i.shape == (batch, B_q, d)

            # TODO
            # 初始化输出
            O_i_j_minus_1 = torch.zeros_like(q_i).to(device)
            # assert O_i_j_minus_1.shape == (batch, B_q, d)
            
            # 初始化log-sum-exp，对于q_i来说
            L_i_j_minus_1 = torch.zeros((batch, B_q)).to(device)
            # assert L_i_j_minus_1.shape == (batch, B_q)

            # 初始化max-value，对于q_i来说
            m_i_j_minus_1 = torch.full((batch, B_q), -float('inf')).to(device)
            # assert m_i_j_minus_1.shape== (batch, B_q)


            # 遍历所有的kv
            for j in range(0, T_k):
                k_j = splited_K[j]
                v_j = splited_V[j]
        
                # assert k_j.shape == (batch, B_k, d)
                # assert v_j.shape == (batch, B_k, d)
                     
                # 计算q_i对于k_j的注意力，使用内积
                S_ij =  q_i @ k_j.transpose(-2, -1) / torch.sqrt(torch.tensor(d, device=q_i.device))
            
                # assert S_ij.shape == (batch, B_q, B_k)
                # m_ij
                row_max_ij= torch.max(S_ij, dim=-1).values
                # assert row_max_ij.shape == (batch, B_q)
                

                m_ij = torch.max(torch.cat([m_i_j_minus_1.unsqueeze(-1), row_max_ij.unsqueeze(-1)], dim=-1), dim=-1).values
                # assert m_ij.shape == (batch, B_q)
                
                Phat_ij = torch.exp(S_ij - m_ij.unsqueeze(-1))
                # assert Phat_ij.shape == (batch, B_q, B_k)
                                
                # TODO
                # print(f"torch.sum(Phat_ij, dim=-1) shape is {torch.sum(Phat_ij, dim=-1).shape}")
                L_ij = torch.exp(m_i_j_minus_1-m_ij) * L_i_j_minus_1 + torch.sum(Phat_ij, dim=-1)
                # assert L_ij.shape == (batch, B_q)
                
                    
                O_ij = torch.diag_embed(torch.exp(m_i_j_minus_1-m_ij)) @ O_i_j_minus_1 + Phat_ij.to(v_j.dtype) @ v_j
                
                # assert O_ij.shape == (batch, B_q, d)
                
                O_i_j_minus_1 = O_ij
                L_i_j_minus_1 = L_ij
                m_i_j_minus_1 = m_ij
            
            

            L_i_tk = L_i_j_minus_1
            O_i_tk = O_i_j_minus_1
            m_i_tk = m_i_j_minus_1


            O_i = torch.inverse(torch.diag_embed(L_i_tk)) @ O_i_tk
            O_i = O_i.to(hidden_embedding_type)

            L_i = m_i_tk + torch.log(L_i_tk)

            O_list.append(O_i)
            L_list.append(L_i)
    
        O = torch.cat(O_list, dim=-2)
        L = torch.cat(L_list, dim=-1)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    
    @staticmethod
    def backward(ctx, grad_O):
        Q, K, V, O, L = ctx.saved_tensors
        d = K.shape[-1]

        scale = 1 / (d ** 0.5)
        with torch.enable_grad():

            S = einsum(Q, K, '... q d, ... k d -> ... q k') * scale
            if ctx.is_causal:
                mask = torch.tril(torch.ones(S.shape[-2], S.shape[-1], device=S.device)) # 下三角为1
                S = S.masked_fill(mask==0, -torch.inf) 
            # 这里实现的时候不知道怎么回事，如果将L.unsqueeze(-1)换成L[:, None]，就会报错
            P = torch.exp(S - L.unsqueeze(-1))

            grad_V = einsum(P, grad_O, '... q k, ... q d -> ... k d')
            grad_P = einsum(grad_O, V, '... q d, ... k d -> ... q k')

            D = torch.sum(O * grad_O, dim=-1)
            grad_S = P * (grad_P - D.unsqueeze(-1))

            grad_Q = einsum(grad_S, K, '... q k, ... k d -> ... q d') * scale    
            grad_K = einsum( grad_S, Q, '... q k, ... q d -> ... k d') * scale

            return grad_Q, grad_K, grad_V, None
