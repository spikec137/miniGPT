# generate.py
import torch
from model import MiniGPT

# 1. 加载训练好的 checkpoint
ckpt = torch.load('checkpoints/mini-gpt.pt', map_location='cpu')

# 2. 从 checkpoint 中提取必要信息
vocab_size = len(ckpt['stoi'])
block_size = ckpt['block_size']
embed_size = ckpt.get('embed_size', 64)       # 默认值用于兼容旧模型
num_layers = ckpt.get('num_layers', 2)        # 默认值用于兼容旧模型
stoi = ckpt['stoi']
itos = ckpt['itos']

# 3. 初始化模型并加载权重
model = MiniGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    embed_size=embed_size,
    num_layers=num_layers
).to('cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 4. 准备初始输入 context（以“你”开头）
context = torch.tensor([[stoi['你']]], dtype=torch.long)

# 5. 生成文本
print("生成文本：")
print(model.generate(context, max_new_tokens=100, stoi=stoi, itos=itos))
