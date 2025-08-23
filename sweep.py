import os
import itertools
import subprocess
import argparse
import csv
import time
import torch

# -------------------------------
# 文本生成函数
# -------------------------------
def run_generation(model_path, output_dir, prompt="", gen_length=100):
    gen_cmd = [
        "python", "generate.py",
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--max_length", str(gen_length)
    ]
    if prompt:
        gen_cmd += ["--prompt", prompt]
    try:
        subprocess.run(gen_cmd, check=True)
        print(f"✅ Generation completed for {model_path}")
    except subprocess.CalledProcessError:
        print(f"⚠️ Generation failed for {model_path}")

# -------------------------------
# 训练 + 自动扩展逻辑
# -------------------------------
def run_training(data_file, params, fast_epochs, max_epochs, min_improve, result_dir, gen_length, prompt):
    folder_name = f"{os.path.splitext(os.path.basename(data_file))[0]}_" \
                  f"bs{params['batch_size']}_emb{params['embed_dim']}_L{params['num_layers']}" \
                  f"_H{params['hidden_dim']}_lr{params['lr']}_dp{params['dropout']}"
    save_dir = os.path.join(result_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, "loss.txt")
    model_path = os.path.join(save_dir, "model.pt")

    epochs = fast_epochs
    best_loss = float("inf")
    patience = 2
    bad_rounds = 0
    max_mem_usage = 0

    while epochs <= max_epochs:
        cmd = [
            "python", "train.py",
            "--data_file", data_file,
            "--save_dir", save_dir,
            "--output_dir", save_dir,
            "--epochs", str(epochs),
            "--batch_size", str(params["batch_size"]),
            "--embed_dim", str(params["embed_dim"]),
            "--num_layers", str(params["num_layers"]),
            "--hidden_dim", str(params["hidden_dim"]),
            "--lr", str(params["lr"]),
            "--dropout", str(params["dropout"])
        ]

        print(f"\n>>> Training: {folder_name}, epochs={epochs}")
        subprocess.run(cmd, check=True)

        # 读取 loss.txt 最新一行
        try:
            with open(log_file, "r") as f:
                last_line = f.readlines()[-1].strip()
                current_loss = float(last_line.split()[-1])
        except Exception:
            print("⚠️ Warning: cannot read loss.txt, stop training")
            break

        # 显存监控
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1024 ** 2
            max_mem_usage = max(max_mem_usage, mem)
            torch.cuda.reset_peak_memory_stats()

        improve = best_loss - current_loss
        if improve > min_improve:
            print(f"✅ Loss improved {improve:.4f}, continue training")
            best_loss = current_loss
            epochs *= 2
            bad_rounds = 0
        else:
            bad_rounds += 1
            print(f"⏹ Loss improvement {improve:.4f} < {min_improve}, bad_rounds={bad_rounds}")
            if bad_rounds >= patience:
                print("⚠️ Early stopping")
                break

    # 训练结束后生成文本
    run_generation(model_path, save_dir, prompt=prompt, gen_length=gen_length)

    return best_loss, model_path, max_mem_usage

# -------------------------------
# 主 sweep 逻辑
# -------------------------------
def main(args):
    param_space = {
        "batch_size": [8, 16, 32, 64],
        "embed_dim": [32, 64, 128, 256],
        "num_layers": [2, 4, 6, 8, 12],
        "hidden_dim": [64, 128, 256, 512],
        "lr": [1e-3, 5e-4, 1e-4],
        "dropout": [0.0, 0.1, 0.2, 0.3]
    }

    keys, values = zip(*param_space.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    param_combinations = sorted(
        param_combinations,
        key=lambda p: (p["embed_dim"], p["num_layers"], p["hidden_dim"], p["batch_size"])
    )

    os.makedirs(args.result_dir, exist_ok=True)
    csv_path = os.path.join(args.result_dir, "sweep_results.csv")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["dataset", "batch_size", "embed_dim", "num_layers",
                         "hidden_dim", "lr", "dropout", "final_loss", "model_path", "max_mem_MB"])

        for data_file in os.listdir(args.data_dir):
            if not data_file.endswith(".txt"):
                continue
            data_path = os.path.join(args.data_dir, data_file)

            for params in param_combinations:
                try:
                    loss, model_path, max_mem = run_training(
                        data_path, params,
                        args.fast_epochs, args.max_epochs,
                        args.min_improve, args.result_dir,
                        args.gen_length, args.prompt
                    )
                    writer.writerow([
                        data_file,
                        params["batch_size"],
                        params["embed_dim"],
                        params["num_layers"],
                        params["hidden_dim"],
                        params["lr"],
                        params["dropout"],
                        loss,
                        model_path,
                        max_mem
                    ])
                    csvfile.flush()
                except subprocess.CalledProcessError:
                    print(f"❌ Training failed for {params}, skipping...")
                    continue

# -------------------------------
# CLI 入口
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="训练数据文件夹")
    parser.add_argument("--result_dir", type=str, default="results", help="结果保存文件夹")
    parser.add_argument("--fast_epochs", type=int, default=64, help="快速预训练 epochs")
    parser.add_argument("--max_epochs", type=int, default=4096, help="最大训练 epochs")
    parser.add_argument("--min_improve", type=float, default=0.0001, help="最小 loss 改善值")
    parser.add_argument("--gen_length", type=int, default=100, help="生成文本长度")
    parser.add_argument("--prompt", type=str, default="", help="生成文本可选 prompt")
    args = parser.parse_args()

    main(args)