# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.key   = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(1, 1, 256, 256)))  # mask

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))  # 避免看到未来
        wei = F.softmax(wei, dim=-1)  # 注意力权重

        out = wei @ v  # (B, T, head_size)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.sa = SelfAttention(embed_size, head_size)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # 残差连接
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, embed_size=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(
            TransformerBlock(embed_size, embed_size),
            TransformerBlock(embed_size, embed_size),
        )
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)  # 输出每个位置预测的下一个 token

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                          # (B, T, C)
        pos = torch.arange(T, device=idx.device)                    # (T,)
        pos_emb = self.position_embedding(pos)[None, :, :]          # (1, T, C)
        x = tok_emb + pos_emb                                       # (B, T, C)
        x = self.blocks(x)                                          # (B, T, C)
        x = self.ln_f(x)                                            # (B, T, C)
        logits = self.head(x)                                       # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, stoi, itos):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -256:]  # 限制最大上下文
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # 只取最后一个位置
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return ''.join([itos[i] for i in idx[0].tolist()])
