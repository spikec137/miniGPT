import argparse
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"  # 解决 MKL+gomp 冲突
import subprocess
import sys
import glob
import csv
import numpy as np
import torch

def read_final_loss(folder):
    loss_file = os.path.join(folder, 'loss.txt')
    if not os.path.exists(loss_file):
        return None
    try:
        losses = np.loadtxt(loss_file, dtype=float, ndmin=1)
    except Exception:
        # 兼容逐行读取
        losses = []
        with open(loss_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    losses.append(float(line.split()[-1]) if ' ' in line else float(line))
                except:
                    pass
        if not losses:
            return None
        losses = np.array(losses, dtype=float)
    if losses.size == 0:
        return None
    return float(losses[-1]), int(losses.size)

def run_train_new(data_file, params, epochs, out_dir):
    """首轮训练：不 resume，直接跑 epochs。"""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "train.py",
        "--data_file", data_file,
        "--output_dir", out_dir,
        "--epochs", str(epochs),
        "--batch_size", str(params["batch_size"]),
        "--num_layers", str(params["num_layers"]),
        "--block_size", str(params["block_size"]),
    ]
    # 兼容你之前用到的别名参数（train.py 会处理优先级，不改变模型结构）
    cmd += [
        "--embed_dim", str(params["embed_dim"]),
        "--hidden_dim", str(params["hidden_dim"]),
        "--lr", str(params["lr"]),
        "--dropout", str(params["dropout"])
    ]
    subprocess.run(cmd, check=True)

def run_train_resume(data_file, params, extra_epochs, out_dir):
    """增量训练：resume + extra_epochs。"""
    if extra_epochs <= 0:
        return
    cmd = [
        sys.executable, "train.py",
        "--data_file", data_file,
        "--output_dir", out_dir,
        "--resume",
        "--extra_epochs", str(extra_epochs),
        "--batch_size", str(params["batch_size"]),
        "--num_layers", str(params["num_layers"]),
        "--block_size", str(params["block_size"]),
    ]
    cmd += [
        "--embed_dim", str(params["embed_dim"]),
        "--hidden_dim", str(params["hidden_dim"]),
        "--lr", str(params["lr"]),
        "--dropout", str(params["dropout"])
    ]
    subprocess.run(cmd, check=True)

def run_generation(model_path, output_dir, prompt, gen_length):
    gen_cmd = [
        sys.executable, "generate.py",
        "--model_path", model_path,
        "--output_dir", output_dir,
        "--max_length", str(gen_length)
    ]
    if prompt:
        gen_cmd += ["--prompt", prompt]
    subprocess.run(gen_cmd, check=True)

def main(args):
    # 搜索 data/*.txt
    data_files = sorted(glob.glob(os.path.join(args.data_dir, "*.txt")))
    os.makedirs(args.result_dir, exist_ok=True)
    csv_path = os.path.join(args.result_dir, "sweep_results.csv")

    # 组合空间（保持你原来的丰富度；从小到大）
    batch_sizes = [8, 16, 32, 64]
    embed_dims  = [32, 64, 128, 256]
    num_layers  = [2, 4, 6, 8, 12]
    block_sizes = [16, 32]  # 你之前固定 16；这里给一个可观察的变化
    lrs         = [1e-3, 5e-4, 1e-4]
    dropouts    = [0.0, 0.1, 0.2, 0.3]
    hidden_dims = [64, 128, 256, 512]

    # 写 CSV 头
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "dataset","batch_size","embed_dim","num_layers","block_size",
                "hidden_dim","lr","dropout","final_loss","trained_epochs","model_path","max_mem_MB"
            ])

        for data_file in data_files:
            data_name = os.path.splitext(os.path.basename(data_file))[0]
            for bs in batch_sizes:
                for emb in embed_dims:
                    for nl in num_layers:
                        for blk in block_sizes:
                            for lr in lrs:
                                for dp in dropouts:
                                    for hd in hidden_dims:
                                        params = {
                                            "batch_size": bs,
                                            "embed_dim": emb,
                                            "num_layers": nl,
                                            "block_size": blk,
                                            "lr": lr,
                                            "dropout": dp,
                                            "hidden_dim": hd
                                        }

                                        out_dir = os.path.join(
                                            args.result_dir,
                                            f"{data_name}_bs{bs}_emb{emb}_L{nl}_B{blk}_lr{lr}_dp{dp}"
                                        )
                                        model_path = os.path.join(out_dir, "model.pt")

                                        print(f"\n=== Fast training: data={data_name}, bs={bs}, emb={emb}, L={nl}, B={blk}, lr={lr}, dp={dp} ===")
                                        try:
                                            # 首轮：fast_epochs
                                            run_train_new(data_file, params, args.fast_epochs, out_dir)
                                        except subprocess.CalledProcessError as e:
                                            print(f"Fast training failed: {e}")
                                            continue

                                        # 读取当前 loss + 已训练总轮数
                                        final_loss, trained_epochs = read_final_loss(out_dir)
                                        if final_loss is None:
                                            print(f"No loss info for {out_dir}, skipping.")
                                            continue

                                        # 记录显存峰值
                                        max_mem = 0.0
                                        if torch.cuda.is_available():
                                            max_mem = torch.cuda.max_memory_allocated() / 1024**2
                                            torch.cuda.reset_peak_memory_stats()

                                        # 若不达标就直接进入“增量训练 + 早停”循环
                                        best_loss = final_loss
                                        current_epochs = trained_epochs
                                        patience = 2
                                        bad_rounds = 0

                                        while current_epochs < args.max_epochs:
                                            target = min(args.max_epochs, current_epochs * 2)
                                            extra = target - current_epochs
                                            if extra <= 0:
                                                break  # 修复 4096 重复问题
                                            print(f"Continue training (resume): +{extra} epochs to reach {target} total.")
                                            try:
                                                run_train_resume(data_file, params, extra, out_dir)
                                            except subprocess.CalledProcessError as e:
                                                print(f"Resume training failed: {e}")
                                                break

                                            # 训练后读取新的 loss 和总轮数
                                            new_loss, new_trained = read_final_loss(out_dir)
                                            if new_loss is None:
                                                break
                                            improvement = best_loss - new_loss
                                            current_epochs = new_trained
                                            best_loss = new_loss

                                            if torch.cuda.is_available():
                                                mem_now = torch.cuda.max_memory_allocated() / 1024**2
                                                max_mem = max(max_mem, mem_now)
                                                torch.cuda.reset_peak_memory_stats()

                                            if improvement < args.min_improve:
                                                bad_rounds += 1
                                                print(f"Loss improvement {improvement:.6f} < min_improve {args.min_improve}, bad_rounds={bad_rounds}")
                                                if bad_rounds >= patience:
                                                    print("Early stopping due to small improvements.")
                                                    break
                                            else:
                                                bad_rounds = 0
                                                print(f"Loss improved by {improvement:.6f}, continue...")

                                        # 生成样例
                                        try:
                                            run_generation(model_path, out_dir, args.prompt, args.gen_length)
                                        except subprocess.CalledProcessError as e:
                                            print(f"Generation failed: {e}")

                                        # 结果落盘
                                        writer.writerow([
                                            data_name, bs, emb, nl, blk, hd, lr, dp,
                                            best_loss, current_epochs, model_path, f"{max_mem:.2f}"
                                        ])
                                        csvfile.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adaptive hyperparameter sweep for MiniGPT (resume training)")
    parser.add_argument("--data_dir", type=str, default="data", help="训练数据文件夹")
    parser.add_argument("--result_dir", type=str, default="results", help="结果保存文件夹")
    parser.add_argument("--fast_epochs", type=int, default=64, help="首轮快速训练 epochs")
    parser.add_argument("--max_epochs", type=int, default=4096, help="最大训练 epochs（总计）")
    parser.add_argument("--min_improve", type=float, default=1e-4, help="继续训练的最小 loss 改善阈值")
    parser.add_argument("--gen_length", type=int, default=100, help="生成文本长度")
    parser.add_argument("--prompt", type=str, default="", help="生成文本可选 prompt")
    args = parser.parse_args()
    main(args)
