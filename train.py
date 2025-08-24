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
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.register_buffer("pos_ids", torch.arange(block_size).unsqueeze(0), persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        pos_emb = self.position_embedding(self.pos_ids[:, :seq_len])
        tok_emb = self.token_embedding(x)
        h = tok_emb + pos_emb
        out = self.transformer(h)
        logits = self.linear(out)
        return logits

def _setup_logger(log_path):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
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
                try:
                    losses.append(float(line))
                except Exception:
                    pass
    return losses

def train_model(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'train.log')
    logger = _setup_logger(log_file)

    embed_size = args.embed_dim if args.embed_dim is not None else args.embed_size
    learning_rate = args.lr if args.lr is not None else args.learning_rate

    with open(args.data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    dataset = CharDataset(text, block_size=args.block_size)
    if len(dataset) == 0:
        logger.error("Dataset is empty after applying block_size, cannot train.")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=min(4, os.cpu_count() or 1),
        persistent_workers=True if torch.cuda.is_available() else False,
        prefetch_factor=2,
        drop_last=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPTModel(dataset.vocab_size, embed_size, args.block_size, args.num_layers).to(device)

    model_file = os.path.join(args.output_dir, 'model.pt')
    total_epochs_done = 0
    best_loss = float('inf')

    if args.resume and os.path.exists(model_file):
        ckpt = torch.load(model_file, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        logger.info(f"Resumed weights from {model_file}")
        if hasattr(ckpt, 'best_loss'):
            best_loss = ckpt['best_loss']

    try:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, foreach=True)
    except TypeError:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    # 判断小文本 vs 大文本
    SMALL_DATASET_THRESHOLD = 5000  # 字符数小于 5000 认为是小文本
    is_small_dataset = len(dataset) < SMALL_DATASET_THRESHOLD
    run_epochs = args.epochs if not args.resume else args.extra_epochs
    patience = args.patience
    bad_rounds = 0

    logger.info(
        f"Starting {'RESUME' if args.resume else 'NEW'} training: "
        f"data={args.data_file}, batch={args.batch_size}, embed={embed_size}, "
        f"layers={args.num_layers}, lr={learning_rate}, block={args.block_size}, "
        f"total_epochs_done={total_epochs_done}, best_loss={best_loss}"
    )

    while total_epochs_done < args.max_epochs:
        model.train()
        total_loss = 0.0
        for epoch in range(1, run_epochs + 1):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_batch)
                loss = criterion(logits.reshape(-1, dataset.vocab_size), y_batch.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        total_epochs_done += run_epochs

        logger.info(f"Epochs {total_epochs_done-run_epochs+1}-{total_epochs_done}, Loss: {avg_loss:.6f}")

        # Early stopping
        improvement = best_loss - avg_loss
        if improvement < args.min_improve:
            bad_rounds += 1
            if bad_rounds >= patience:
                logger.info("Early stopping: no sufficient improvement.")
                break
        else:
            best_loss = avg_loss
            bad_rounds = 0

        # 小文本直接结束或大文本翻倍
        if is_small_dataset:
            if improvement < args.min_improve:
                logger.info("Small dataset, stopping training early due to no improvement.")
                break
        else:
            run_epochs = min(run_epochs*2, args.max_epochs - total_epochs_done)
            if run_epochs <= 0:
                break

    # 保存模型
    save_data = {
        'model_state': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'stoi': dataset.stoi,
        'itos': dataset.itos,
        'embed_size': embed_size,
        'num_layers': args.num_layers,
        'block_size': args.block_size,
        'best_loss': best_loss
    }
    torch.save(save_data, model_file)
    logger.info(f"Saved final model to {model_file}")

    # 保存 loss
    loss_txt = os.path.join(args.output_dir, 'loss.txt')
    with open(loss_txt, 'w', encoding='utf-8') as f:
        f.write(f"{best_loss}\n")

    # 绘制 loss 图
    plt.figure()
    plt.plot([best_loss], marker='o')
    plt.xlabel('Final')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss.png'))

    logger.info("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MiniGPT with incremental epochs")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--block_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--extra_epochs', type=int, default=5)
    parser.add_argument('--max_epochs', type=int, default=1024)
    parser.add_argument('--min_improve', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    args = parser.parse_args()
    train_model(args)
