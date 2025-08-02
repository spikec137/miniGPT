# dataset.py
import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, text, block_size):
        self.text = text
        self.block_size = block_size

        # 获取字符集（不重复）
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # 字符 -> 索引
        self.itos = {i: ch for ch, i in self.stoi.items()}  # 索引 -> 字符
        self.vocab_size = len(chars)

        # 编码整个文本为整数列表
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        # 可生成样本的个数 = 文本长度 - 一个训练样本长度
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # 截取一段长度 block_size 作为输入，后一位作为目标
        x = self.data[idx : idx + self.block_size]      # 输入序列
        y = self.data[idx + 1 : idx + 1 + self.block_size]  # 目标序列（向右移一位）
        return x, y

# 用于加载文本并创建数据集对象
def load_dataset(path, block_size):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return CharDataset(text, block_size)
