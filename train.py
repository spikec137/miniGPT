# train.py
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from model import MiniGPT
from dataset import load_dataset

# 超参数
# batch_size = 16
# block_size = 64
batch_size = 4 #一次训练“喂给模型”的样本数量（有点像多少碗饭一起端给模型吃）
block_size = 8 #每个样本的输入序列长度（模型一次能看到多少上下文字符）
max_iters = 1000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载数据集
train_dataset = load_dataset("data/tiny.txt", block_size)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
vocab_size = train_dataset.vocab_size
stoi = train_dataset.stoi
itos = train_dataset.itos

# 初始化模型
model = MiniGPT(vocab_size=vocab_size, block_size=block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 训练循环
for step in range(max_iters):
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break  # 每轮只取一批数据（简化训练）

    # 每隔 eval_interval 输出一次 loss
    if step % eval_interval == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

# 保存模型
ckpt = {
    'model_state_dict': model.state_dict(),
    'stoi': stoi,
    'itos': itos,
    'block_size': block_size,
}
torch.save(ckpt, 'checkpoints/mini-gpt.pt')

# 测试生成
model.eval()
context = torch.tensor([[stoi['你']]], dtype=torch.long).to(device)
print("生成文本：")
print(model.generate(context, max_new_tokens=100, stoi=stoi, itos=itos))
