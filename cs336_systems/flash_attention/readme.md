

# 高效attention的实现

## 文件结构
- attention_torch.py是最基本的pytorch实现的attention前向传播；
- flash_attention_torch.py 是使用torch实现的flash算法前后向传播代码；
- flash_attention_triton是使triton实现的flashattention2算法前后向传播代码；
- dummy_tensor_generation.py:debug： torch实现对算法后，固定qkv的输入参数




