import os
import csv
import itertools
import subprocess
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

# ==== 参数配置 ====
# datasets = ["data/tiny.txt"]
datasets = ["data/StrayBirds.txt"]
batch_sizes = [8, 16]
embed_dims = [32, 64]
num_layers_list = [2]
hidden_dims = [64, 128]
lrs = [0.001]
dropouts = [0.0, 0.1]

max_epochs_limit = 4096  # 最大允许轮数
sweep_csv = "sweep_result.csv"

def save_params_json(output_dir, params):
    """每次训练保存参数到 output_dir/params.json"""
    params_file = os.path.join(output_dir, "params.json")
    with open(params_file, "w") as f:
        json.dump(params, f, indent=4)

def train_model(params, initial_epochs=32):
    """训练模型，支持动态增量翻倍 + 每轮保存参数json"""
    dataset, batch_size, embed_dim, num_layers, hidden_dim, lr, dropout = params

    model_name = f"{os.path.basename(dataset).split('.')[0]}_bs{batch_size}_emb{embed_dim}_L{num_layers}_H{hidden_dim}_lr{lr}_dp{dropout}"
    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)

    total_epochs = 0
    current_epochs = initial_epochs
    prev_loss = float("inf")

    # 训练循环
    while total_epochs < max_epochs_limit:
        logging.info(f"Starting training: batch={batch_size}, embed={embed_dim}, layers={num_layers}, block=16, lr={lr}, dropout={dropout}, epochs={current_epochs}")

        # 检查是否已有模型和 loss.txt
        resume = os.path.exists(os.path.join(output_dir, "model.pt"))

        cmd = [
            "python", "train.py",
            "--data_file", dataset,
            "--output_dir", output_dir,
            "--batch_size", str(batch_size),
            "--embed_dim", str(embed_dim),
            "--hidden_dim", str(hidden_dim),
            "--num_layers", str(num_layers),
            "--lr", str(lr),
            "--dropout", str(dropout),
            "--epochs", str(current_epochs)
        ]
        if resume:
            cmd.append("--resume")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(result.stderr)
            break
        logging.info(result.stdout)

        # 读取最后一条 loss
        loss_file = os.path.join(output_dir, "loss.txt")
        if os.path.exists(loss_file):
            with open(loss_file, "r") as f:
                losses = [float(x.strip()) for x in f.readlines() if x.strip()]
            current_loss = losses[-1]
        else:
            current_loss = float("inf")

        # 保存参数 json
        param_record = {
            "dataset": dataset,
            "batch_size": batch_size,
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "dropout": dropout,
            "current_loss": current_loss,
            "total_epochs": total_epochs + current_epochs,
            "model_path": os.path.join(output_dir, "model.pt")
        }
        save_params_json(output_dir, param_record)

        # Early stopping 判定
        if prev_loss - current_loss < 1e-6:
            logging.info("Early stopping: no sufficient improvement.")
            break
        prev_loss = current_loss

        total_epochs += current_epochs
        current_epochs = min(current_epochs * 2, max_epochs_limit - total_epochs)
        logging.info(f"Loss improved, continue training... next epochs={current_epochs}")

    return param_record

def generate_text(model_path, output_dir, prompt="Hello", max_length=100):
    cmd = [
        "python", "generate.py",
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--prompt", prompt,
        "--max_length", str(max_length)
    ]
    subprocess.run(cmd)

def main():
    fieldnames = ["dataset","batch_size","embed_dim","num_layers","hidden_dim","lr","dropout","final_loss","model_path"]
    if not os.path.exists(sweep_csv):
        with open(sweep_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    param_combinations = list(itertools.product(datasets, batch_sizes, embed_dims, num_layers_list, hidden_dims, lrs, dropouts))
    for params in param_combinations:
        logging.info(f"=== Training: {params} ===")
        record = train_model(params)

        # 写入 sweep_result.csv
        with open(sweep_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "dataset": record["dataset"],
                "batch_size": record["batch_size"],
                "embed_dim": record["embed_dim"],
                "num_layers": record["num_layers"],
                "hidden_dim": record["hidden_dim"],
                "lr": record["lr"],
                "dropout": record["dropout"],
                "final_loss": record["current_loss"],
                "model_path": record["model_path"]
            })

        # 调用生成文本
        generate_text(record["model_path"], os.path.dirname(record["model_path"]))

if __name__ == "__main__":
    main()
