# train.py

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import MiniGPT
from dataset import load_dataset

# 超参数配置
batch_size = 8           # 每批次输入样本数量
block_size = 8           # 序列长度（上下文长度）
max_iters = 5000         # 训练步数
eval_interval = 100      # 每隔多少步打印一次 loss
learning_rate = 1e-3     # 学习率
embed_size = 128          # 每个 token 表示成多长的向量
num_layers = 12          # Transformer 层数
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载数据集
# train_dataset = load_dataset("data/tiny.txt", block_size)
train_dataset = load_dataset("data/StrayBirds.txt", block_size)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
vocab_size = train_dataset.vocab_size
stoi = train_dataset.stoi
itos = train_dataset.itos

# 初始化模型
model = MiniGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    embed_size=embed_size,
    num_layers=num_layers
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 记录 loss 曲线
losses = []

# 训练循环
for step in range(max_iters):
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        break  # 每步只训练一批数据（简单快速训练）

    # 每 eval_interval 步打印一次 loss
    if step % eval_interval == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
        losses.append((step, loss.item()))

# 保存 loss 曲线图
if losses:
    steps, loss_vals = zip(*losses)
    plt.plot(steps, loss_vals, marker='o')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    print("✅ Loss 曲线保存为 loss_curve.png")

# 保存模型
ckpt = {
    'model_state_dict': model.state_dict(),
    'stoi': stoi,
    'itos': itos,
    'block_size': block_size,
}
torch.save(ckpt, 'checkpoints/mini-gpt.pt')
print("模型已保存到 checkpoints/mini-gpt.pt")

# 测试生成
model.eval()
context = torch.tensor([[stoi['你']]], dtype=torch.long).to(device)
print("生成文本：")
print(model.generate(context, max_new_tokens=100, stoi=stoi, itos=itos))

ckpt = {
    'model_state_dict': model.state_dict(),
    'stoi': stoi,
    'itos': itos,
    'block_size': block_size,
    'embed_size': embed_size,
    'num_layers': num_layers
}
torch.save(ckpt, 'checkpoints/mini-gpt.pt')
