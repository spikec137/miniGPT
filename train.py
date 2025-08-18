import argparse
import os
import logging
import sys
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
    """简易Transformer语言模型"""
    def __init__(self, vocab_size, embed_size, num_layers, block_size, nhead=2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(pos_ids)
        h = tok_emb + pos_emb
        h = h.permute(1, 0, 2)
        out = self.transformer(h)
        out = out.permute(1, 0, 2)
        logits = self.linear(out)
        return logits

def train_model(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'train.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"Starting training: batch={args.batch_size}, embed={args.embed_size}, layers={args.num_layers}, lr={args.learning_rate}")

    # 读取数据集
    with open(args.data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    dataset = CharDataset(text, block_size=args.block_size)
    if len(dataset) == 0:
        logger.error("Dataset is empty after applying block_size, cannot train.")
        return
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = GPTModel(dataset.vocab_size, args.embed_size, args.num_layers, args.block_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            logits = logits.view(-1, dataset.vocab_size)
            targets = y_batch.view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}")

    # 保存模型
    model_file = os.path.join(args.output_dir, 'model.pt')
    save_data = {
        'model_state': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'stoi': dataset.stoi,
        'itos': dataset.itos,
        'embed_size': args.embed_size,
        'num_layers': args.num_layers,
        'block_size': args.block_size
    }
    torch.save(save_data, model_file)
    logger.info(f"Saved model to {model_file}")

    # 保存 loss.txt
    loss_txt = os.path.join(args.output_dir, 'loss.txt')
    with open(loss_txt, 'w', encoding='utf-8') as f:
        for l in losses:
            f.write(f"{l}\n")
    logger.info(f"Saved loss data to {loss_txt}")

    # 绘制 loss 曲线
    plt.figure()
    plt.plot(range(1, args.epochs + 1), losses, marker='o')
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
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--embed_size',    type=int,   default=64)
    parser.add_argument('--num_layers',    type=int,   default=2)
    parser.add_argument('--block_size',    type=int,   default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs',        type=int,   default=5)
    parser.add_argument('--output_dir',    type=str,   default='.')
    parser.add_argument('--data_file',     type=str, required=True, help='Path to data file')
    args = parser.parse_args()
    train_model(args)