import argparse
import os
import logging
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
import threading

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
    if args.resume and os.path.exists(model_file):
        ckpt = torch.load(model_file, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        logger.info(f"Resumed weights from {model_file}")

    try:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, foreach=True)
    except TypeError:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    total_epochs_done = 0
    best_loss = float('inf')
    bad_rounds = 0

    # 超时 flag
    stop_flag = {'single_epoch': False, 'total': False}

    # 记录每轮 loss + 时间
    loss_time_file = os.path.join(args.output_dir, 'loss_time.json')
    loss_time_records = []

    # 启动总训练时间计时器线程
    def total_timer():
        time.sleep(args.max_total_time)
        stop_flag['total'] = True

    threading.Thread(target=total_timer, daemon=True).start()
    start_total_time = time.time()

    while total_epochs_done < args.max_epochs:
        run_epochs = min(args.epochs, args.max_epochs - total_epochs_done)
        if run_epochs <= 0:
            break

        # 启动单轮 epoch 计时器
        def single_epoch_timer():
            time.sleep(args.max_single_epoch_time)
            stop_flag['single_epoch'] = True

        threading.Thread(target=single_epoch_timer, daemon=True).start()
        epoch_start_time = time.time()
        total_loss = 0.0

        for _ in range(run_epochs):
            for x_batch, y_batch in loader:
                if stop_flag['single_epoch'] or stop_flag['total']:
                    break
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_batch)
                loss = criterion(logits.reshape(-1, dataset.vocab_size), y_batch.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if stop_flag['single_epoch'] or stop_flag['total']:
                break

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        avg_loss = total_loss / len(loader)
        total_epochs_done += run_epochs

        loss_time_records.append({'epochs_done': total_epochs_done, 'loss': avg_loss, 'epoch_time_sec': epoch_time})
        with open(loss_time_file, 'w', encoding='utf-8') as f:
            json.dump(loss_time_records, f, indent=2)

        logger.info(f"Epochs {total_epochs_done-run_epochs+1}-{total_epochs_done}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")

        if stop_flag['single_epoch']:
            logger.info(f"Single epoch exceeded {args.max_single_epoch_time}s, stopping this param set.")
            break
        if stop_flag['total']:
            logger.info(f"Total training time exceeded {args.max_total_time}s, stopping this param set.")
            break

        # Early stopping
        improvement = best_loss - avg_loss
        if improvement < args.min_improve:
            bad_rounds += 1
            if bad_rounds >= args.patience:
                logger.info("Early stopping: no sufficient improvement.")
                break
        else:
            best_loss = avg_loss
            bad_rounds = 0

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
    logger.info(f"Saved final model to {model_file}")

    # 保存最终 loss 到 loss.txt
    loss_txt = os.path.join(args.output_dir, 'loss.txt')
    with open(loss_txt, 'w', encoding='utf-8') as f:
        f.write(f"{best_loss}\n")

    # 绘制 loss 曲线
    plt.figure()
    plt.plot([r['loss'] for r in loss_time_records], marker='o')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss.png'))
    logger.info("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MiniGPT with incremental epochs and time control")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--block_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
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
    parser.add_argument("--max_single_epoch_time", type=float, default=120)  # 单轮最大时间秒
    parser.add_argument("--max_total_time", type=float, default=1800)        # 总训练最大时间秒
    args = parser.parse_args()
    train_model(args)
