import triton
import triton.language as tl
import torch
import torch.nn as nn
from einops import einsum, rearrange






class backward_recompute_safe_exp(nn.Module):
    def __init__(self):
        super(backward_recompute_safe_exp, self).__init__()
        pass

    def forward(self, Q, K, V, L, is_causal=False):
        d = Q.shape[-1]
        scale = 1 / (d ** 0.5)
        S = einsum(Q, K, '... q d, ... k d -> ... q k') * scale
        if is_causal:
            mask = torch.tril(torch.ones(S.shape[-2], S.shape[-1], device=S.device)) # 下三角为1
            S = S.masked_fill(mask==0, -torch.inf) 
        P = torch.exp(S -L.unsqueeze(-1))
        return P

    
class NiuFlashAttentionV2_triton(torch.autograd.Function): 
    
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        
        
        batch, seq_len, d = Q.shape
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        scale = 1 / (d ** 0.5)
        # 申请存放O和log-sum-exp的内存
        O = torch.empty((batch, seq_len, d), device=Q.device, dtype=Q.dtype)

        # log-sum-exp 因为精度要求高，所以用float32
        L = torch.empty((batch, seq_len), device=Q.device, dtype=torch.float32)
        
        grid = (triton.cdiv(seq_len, Q_TILE_SIZE), batch)
        
        flash_fwd_kernel[grid](Q, K, V, O, L,
                               Q.stride(0), Q.stride(1), Q.stride(2),
                               K.stride(0), K.stride(1), K.stride(2),
                               V.stride(0), V.stride(1), V.stride(2),
                               O.stride(0), O.stride(1), O.stride(2),
                               L.stride(0), L.stride(1),
                               seq_len, seq_len,
                               scale,
                               d,
                               Q_TILE_SIZE,
                               K_TILE_SIZE,
                               is_causal)
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, grad_O):

        Q, K, V, O, L = ctx.saved_tensors
        d = Q.shape[-1]
        scale = 1 / (d ** 0.5)
        with torch.enable_grad():

            # 这里用你之前写的 _attention_and_lse（PyTorch 版）
            # recompute safe exp
            P = backward_recompute_safe_exp()(Q, K, V, L, is_causal=ctx.is_causal)


            grad_V = einsum(P, grad_O, '... q k, ... q d -> ... k d')
            grad_P = einsum(grad_O, V, '... q d, ... k d -> ... q k')

            # 计算D向量
            D = torch.sum(O * grad_O, dim=-1)

            grad_S = P * (grad_P - D.unsqueeze(-1))

            grad_Q = einsum(grad_S, K, '... q k, ... k d -> ... q d') * scale

            grad_K = einsum(grad_S, Q, '... q k, ... q d -> ... k d') * scale

            return grad_Q, grad_K, grad_V, None
    


    
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    
    
    # on chip 初始化一些中间变量
    O_i_j_minus_1 = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    L_i_j_minus_1 = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i_j_minus_1 = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)
    
    q_i = tl.load(Q_block_ptr).to(tl.float32)
    # tl.device_assert(q_i.dtype == tl.float32, "Q must be float32!")


    T_k = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(0, T_k):
        
        # assert k_j.shape == (K_TILE_SIZE, D)
        # assert v_j.shape == (K_TILE_SIZE, D)
        k_j = tl.load(K_block_ptr).to(tl.float32)
        v_j = tl.load(V_block_ptr)
    
        
        # 计算注意力分
        S_ij = tl.dot(q_i, tl.trans(k_j)) * scale
        # tl.device_print("S_ij", S_ij)
        # tl.device_print("S_ij", S_ij)

        if is_causal:
            offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            offs_k = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            # 合法位置：query 位置 >= key 位置
            valid_mask = offs_q[:, None] >= offs_k[None, :]
            # 只保留合法位置的分数，非法位置设为 -inf
            S_ij = tl.where(valid_mask, S_ij, S_ij + float('-inf'))
        
        
        # 求出这一小块的最大值
        row_max_ij = tl.max(S_ij, axis=-1)
        
        
        # m_ij shape is (Q_TILE_SIZE,)
        m_ij = tl.maximum(m_i_j_minus_1, row_max_ij)
        
        # Phat_ij shape is (Q_TILE_SIZE, K_TILE_SIZE)
        Phat_ij = tl.exp(S_ij - m_ij[:, None])
        # L_ij.shape == (Q_TILE_SIZE,)
        L_ij = tl.exp(m_i_j_minus_1-m_ij) * L_i_j_minus_1 + tl.sum(Phat_ij, axis=-1)
        # O_ij shape is (Q_TILE_SIZE, D)
        O_ij = tl.exp(m_i_j_minus_1- m_ij)[:, None] * O_i_j_minus_1 + tl.dot(Phat_ij.to(V_block_ptr.type.element_ty), v_j)
        
            
        O_i_j_minus_1 = O_ij
        L_i_j_minus_1 = L_ij
        m_i_j_minus_1 = m_ij
    
        # 移动到下一个tile
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
        
        
        
    O_i = O_i_j_minus_1  / L_i_j_minus_1[:, None]
    
    
                
    L_i = m_i_j_minus_1 + tl.log(L_i_j_minus_1) # L_i shape is (Q_TILE_SIZE,)
    # if batch_index == 0 and query_tile_index == 0:
    #     tl.device_print("L_i", L_i)  # pyright: ignore[reportUnreachable]
    
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i)
    
    
        
        
        


