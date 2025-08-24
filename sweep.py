import os
import itertools
import logging
import subprocess
import json
import time
import torch

# ==================== 配置日志 ====================
log_file = "results/sweep.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logger = logging.getLogger("sweep")
logger.setLevel(logging.INFO)
if logger.handlers:
    logger.handlers.clear()
fmt = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
fh = logging.FileHandler(log_file)
fh.setFormatter(fmt)
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(ch)

# ==================== 超参数搜索空间 ====================
datasets = ["data/tiny.txt", "data/StrayBirds.txt"]
batch_sizes = [8, 16, 32, 64]
embed_dims = [32, 64, 128, 256]
num_layers_list = [2, 4, 6, 8, 12]
hidden_dims = [64, 128, 256, 512]
lrs = [0.001, 0.0005, 0.0001]
dropouts = [0.0, 0.1, 0.2, 0.3]

# 最大单轮和总训练时间，可覆盖 train.py 默认
MAX_SINGLE_EPOCH_TIME = 60  # 秒
MAX_TOTAL_TIME = 1800        # 秒

# 保存结果 CSV
SWEEP_CSV = "results/sweep_result.csv"
if not os.path.exists(SWEEP_CSV):
    with open(SWEEP_CSV, "w", encoding="utf-8") as f:
        f.write("dataset,batch_size,embed_dim,num_layers,hidden_dim,lr,dropout,final_loss,total_time_sec,avg_epoch_time_sec,model_path,sample_text_path\n")

# ==================== 文本生成函数 ====================
def generate_sample(model_path, data_file, max_len=200, start_str="T"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = checkpoint['vocab_size']
    embed_size = checkpoint['embed_size']
    block_size = checkpoint['block_size']
    num_layers = checkpoint['num_layers']

    # 简单 GPT 架构复现（与 train.py 相同）
    import torch.nn as nn
    class GPTModel(nn.Module):
        def __init__(self, vocab_size, embed_size, block_size, num_layers):
            super().__init__()
            self.block_size = block_size
            self.token_embedding = nn.Embedding(vocab_size, embed_size)
            self.position_embedding = nn.Embedding(block_size, embed_size)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=2, batch_first=True)
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

    model = GPTModel(vocab_size, embed_size, block_size, num_layers).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 过滤掉不在训练字符表的起始字符
    filtered_start = [c for c in start_str if c in stoi]
    if len(filtered_start) == 0:
        # fallback 使用训练文本开头第一个字符
        with open(data_file, 'r', encoding='utf-8') as f:
            text = f.read()
        filtered_start = [text[0]]
    input_ids = torch.tensor([stoi[c] for c in filtered_start], dtype=torch.long).unsqueeze(0).to(device)

    generated = input_ids.tolist()[0]

    for _ in range(max_len):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated.append(next_id.item())
        input_ids = torch.cat([input_ids, next_id], dim=1)
        if input_ids.size(1) > block_size:
            input_ids = input_ids[:, -block_size:]

    sample_text = "".join([itos[i] for i in generated])
    return sample_text

# ==================== 训练函数 ====================
def train_model(params):
    dataset, batch_size, embed_dim, num_layers, hidden_dim, lr, dropout = params
    param_str = f"{os.path.basename(dataset).split('.')[0]}_bs{batch_size}_emb{embed_dim}_L{num_layers}_H{hidden_dim}_lr{lr}_dp{dropout}"
    output_dir = os.path.join("results", param_str)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"=== Training: {params} ===")
    cmd = [
        "python", "train.py",
        "--data_file", dataset,
        "--batch_size", str(batch_size),
        "--num_layers", str(num_layers),
        "--block_size", "16",
        "--embed_dim", str(embed_dim),
        "--hidden_dim", str(hidden_dim),
        "--lr", str(lr),
        "--dropout", str(dropout),
        "--epochs", "32",
        "--max_epochs", "1024",
        "--output_dir", output_dir,
        "--max_single_epoch_time", str(MAX_SINGLE_EPOCH_TIME),
        "--max_total_time", str(MAX_TOTAL_TIME)
    ]
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed for {params}: {e}")
        return None

    total_time = time.time() - start_time

    # 读取 train.py 生成的 loss_time.json
    loss_time_file = os.path.join(output_dir, "loss_time.json")
    if os.path.exists(loss_time_file):
        with open(loss_time_file, "r", encoding="utf-8") as f:
            loss_records = json.load(f)
        final_loss = loss_records[-1]['loss']
        avg_epoch_time = sum(r['epoch_time_sec'] for r in loss_records) / len(loss_records)
    else:
        final_loss = None
        avg_epoch_time = None

    model_path = os.path.join(output_dir, "model.pt")

    # 生成文本
    sample_text = generate_sample(model_path, dataset)
    sample_path = os.path.join(output_dir, "sample.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "embed_dim": embed_dim,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "dropout": dropout,
        "final_loss": final_loss,
        "total_time_sec": total_time,
        "avg_epoch_time_sec": avg_epoch_time,
        "model_path": model_path,
        "sample_text_path": sample_path
    }

# ==================== 主循环 ====================
def main():
    all_params = list(itertools.product(datasets, batch_sizes, embed_dims, num_layers_list, hidden_dims, lrs, dropouts))
    for params in all_params:
        record = train_model(params)
        if record is None:
            continue
        # 追加到 CSV
        with open(SWEEP_CSV, "a", encoding="utf-8") as f:
            f.write(",".join([
                str(record["dataset"]),
                str(record["batch_size"]),
                str(record["embed_dim"]),
                str(record["num_layers"]),
                str(record["hidden_dim"]),
                str(record["lr"]),
                str(record["dropout"]),
                str(record["final_loss"]),
                str(record["total_time_sec"]),
                str(record["avg_epoch_time_sec"]),
                str(record["model_path"]),
                str(record["sample_text_path"])
            ]) + "\n")
        logger.info(f"Finished sweep for {params}, final_loss={record['final_loss']:.6f}, total_time={record['total_time_sec']:.2f}s, sample saved to {record['sample_text_path']}")

if __name__ == "__main__":
    main()
