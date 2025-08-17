import argparse
import os
import subprocess
import sys

def main(args):
    # 定义超参数列表
    batch_sizes   = args.batch_sizes
    embed_sizes   = args.embed_sizes
    num_layers_list = args.num_layers_list
    learning_rates = args.learning_rates
    block_size = args.block_size
    # 遍历所有组合
    for bs in batch_sizes:
        for emb in embed_sizes:
            for nl in num_layers_list:
                for lr in learning_rates:
                    # 构造输出文件夹名称
                    folder_name = os.path.join("results", f"bs{bs}_emb{emb}_L{nl}_B{block_size}_lr{lr}")
                    os.makedirs(folder_name, exist_ok=True)
                    print(f"Training combination: batch={bs}, embed={emb}, layers={nl}, lr={lr}")
                    # 调用 train.py 进行训练
                    train_cmd = [
                        sys.executable, "train.py",
                        "--batch_size", str(bs),
                        "--embed_size", str(emb),
                        "--num_layers", str(nl),
                        "--block_size", str(block_size),
                        "--learning_rate", str(lr),
                        "--epochs", str(args.epochs),
                        "--output_dir", folder_name
                    ]
                    subprocess.run(train_cmd, check=True)
                    # 训练完成后调用 generate.py 生成文本
                    model_path = os.path.join(folder_name, 'model.pt')
                    gen_cmd = [
                        sys.executable, "generate.py",
                        "--model_path", model_path,
                        "--output_dir", folder_name,
                        "--max_length", str(args.gen_length)
                    ]
                    if args.prompt:
                        gen_cmd += ["--prompt", args.prompt]
                    subprocess.run(gen_cmd, check=True)
                    print(f"Completed combination: {folder_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for MiniGPT training")
    parser.add_argument('--batch_sizes',   type=int, nargs='+', default=[16,32],    help='List of batch sizes')
    parser.add_argument('--embed_sizes',   type=int, nargs='+', default=[64,128],   help='List of embedding sizes')
    parser.add_argument('--num_layers_list', type=int, nargs='+', default=[2,4],   help='List of number of layers')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[0.001,0.0005], help='List of learning rates')
    parser.add_argument('--block_size',    type=int, default=16,    help='Block (sequence) size, constant for sweep')
    parser.add_argument('--epochs',        type=int, default=5,     help='Number of epochs for each run')
    parser.add_argument('--gen_length',    type=int, default=100,   help='Length of text to generate')
    parser.add_argument('--prompt',        type=str, default='',     help='Optional prompt for generation')
    args = parser.parse_args()
    main(args)
