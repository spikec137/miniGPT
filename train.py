import argparse
import os
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 让 matmul 用 TF32（非 AMP），Ampere(如 4070)上可提速
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

class CharDataset(Dataset):
    """字符级数据集：将文本转换为滑动窗口的输入-目标对。"""
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(chars)
        data = [self.stoi[c] for c in text]
        self.block_size = block_size
        self.examples = []
        for i in range(len(data) - block_size):
            x = data[i:i+block_size]
            y = data[i+1:i+block_size+1]
            self.examples.append((x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

class GPTModel(nn.Module):
    """简易Transformer语言模型（batch_first=True 去掉额外 permute 开销）"""
    def __init__(self, vocab_size, embed_size, block_size, num_layers, nhead=2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        # 直接 batch_first 提升效率，且消除你之前的 warning
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
        # 预注册位置索引缓冲，避免每步重复创建
        self.register_buffer("pos_ids", torch.arange(block_size).unsqueeze(0), persistent=False)

    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        pos_emb = self.position_embedding(self.pos_ids[:, :seq_len])   # (1, seq_len, embed)
        tok_emb = self.token_embedding(x)                               # (batch, seq_len, embed)
        h = tok_emb + pos_emb                                           # (batch, seq_len, embed)
        out = self.transformer(h)                                       # (batch, seq_len, embed)
        logits = self.linear(out)                                       # (batch, seq_len, vocab)
        return logits

def _setup_logger(log_path):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    # 避免重复添加 handler
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def _read_all_losses(loss_txt):
    if not os.path.exists(loss_txt):
        return []
    losses = []
    with open(loss_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                losses.append(float(line.split()[-1]) if ' ' in line else float(line))
            except ValueError:
                # 容错：如果之前写了花样格式，尝试取末尾数字
                try:
                    losses.append(float(line))
                except Exception:
                    pass
    return losses

def train_model(args):
    # 允许 sweep 里用 --save_dir 覆盖 --output_dir
    if args.save_dir:
        args.output_dir = args.save_dir
    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, 'train.log')
    logger = _setup_logger(log_file)

    # 兼容两个命名：embed_size/learning_rate 与 embed_dim/lr
    embed_size = args.embed_dim if args.embed_dim is not None else args.embed_size
    learning_rate = args.lr if args.lr is not None else args.learning_rate

    # 读取数据
    with open(args.data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    dataset = CharDataset(text, block_size=args.block_size)
    if len(dataset) == 0:
        logger.error("Dataset is empty after applying block_size, cannot train.")
        sys.exit(1)

    # DataLoader：更高效
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=min(4, os.cpu_count() or 1),
        persistent_workers=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if torch.cuda.is_available() else 2,
        # drop_last=True //跳过不足一个 batch 的情况，导致 dataset 太小时直接没 batch,len(loader)=0,报错
        drop_last = False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建模型
    model = GPTModel(dataset.vocab_size, embed_size, args.block_size, args.num_layers).to(device)

    # 如果 resume，则加载已有权重
    model_file = os.path.join(args.output_dir, 'model.pt')
    if args.resume and os.path.exists(model_file):
        ckpt = torch.load(model_file, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        logger.info(f"Resumed weights from {model_file}")

    # 优化器（保持 Adam，不用 AMP；用 foreach 提升效率）
    try:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, foreach=True)
    except TypeError:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    # 训练轮数：首次训练用 epochs；增量训练用 extra_epochs
    run_epochs = args.extra_epochs if args.resume else args.epochs
    if run_epochs <= 0:
        logger.info("No epochs to run (run_epochs<=0).")
        return

    logger.info(
        f"Starting {'RESUME' if args.resume else 'NEW'} training: "
        f"data={args.data_file}, batch={args.batch_size}, embed={embed_size}, "
        f"layers={args.num_layers}, lr={learning_rate}, block={args.block_size}, epochs={run_epochs}"
        + (f", hidden_dim={args.hidden_dim}, dropout={args.dropout}" if args.hidden_dim or args.dropout is not None else "")
    )

    losses_this_run = []
    for epoch in range(1, run_epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)                         # (B,T,V)
            loss = criterion(logits.reshape(-1, dataset.vocab_size), y_batch.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses_this_run.append(avg_loss)
        logger.info(f"Epoch {epoch}/{run_epochs}, Loss: {avg_loss:.4f}")

    # 保存模型
    save_data = {
        'model_state': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'stoi': dataset.stoi,
        'itos': dataset.itos,
        'embed_size': embed_size,
        'num_layers': args.num_layers,
        'block_size': args.block_size
    }
    torch.save(save_data, model_file)
    logger.info(f"Saved model to {model_file}")

    # 处理 loss.txt（追加或新建）
    loss_txt = os.path.join(args.output_dir, 'loss.txt')
    mode = 'a' if args.resume and os.path.exists(loss_txt) else 'w'
    with open(loss_txt, mode, encoding='utf-8') as f:
        for l in losses_this_run:
            f.write(f"{l}\n")
    logger.info(f"Saved loss data to {loss_txt}")

    # 重新读取全量 loss 绘图
    all_losses = _read_all_losses(loss_txt)
    if all_losses:
        plt.figure()
        plt.plot(range(1, len(all_losses)+1), all_losses, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        loss_plot = os.path.join(args.output_dir, 'loss.png')
        plt.savefig(loss_plot)
        logger.info(f"Saved loss plot to {loss_plot}")

    logger.info("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MiniGPT model")
    # 原有参数
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--embed_size',    type=int,   default=64)
    parser.add_argument('--num_layers',    type=int,   default=2)
    parser.add_argument('--block_size',    type=int,   default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs',        type=int,   default=5)
    parser.add_argument('--output_dir',    type=str,   default='.')
    parser.add_argument('--data_file',     type=str,   required=True, help='Path to data file')

    # 你 sweep 里用到但原模型不严格依赖的参数（为了兼容传参）
    parser.add_argument("--save_dir", type=str, default="", help="(optional) alias for output_dir")
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # 新增：增量训练开关
    parser.add_argument("--resume", action="store_true", help="Continue training from existing model.pt in output_dir")
    parser.add_argument("--extra_epochs", type=int, default=0, help="Extra epochs to run when --resume is set")

    args = parser.parse_args()
    train_model(args)
