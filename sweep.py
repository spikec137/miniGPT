import argparse
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import subprocess
import sys
import glob
import numpy as np

def read_final_loss(folder):
    loss_file = os.path.join(folder, 'loss.txt')
    if not os.path.exists(loss_file):
        return None
    losses = np.loadtxt(loss_file)
    if len(losses) == 0:
        return None
    return losses[-1]

def run_training(data_file, bs, emb, nl, block_size, lr, epochs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_cmd = [
        sys.executable, "train.py",
        "--batch_size", str(bs),
        "--embed_size", str(emb),
        "--num_layers", str(nl),
        "--block_size", str(block_size),
        "--learning_rate", str(lr),
        "--epochs", str(epochs),
        "--data_file", data_file,
        "--output_dir", output_dir
    ]
    subprocess.run(train_cmd, check=True)

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
    data_files = glob.glob(os.path.join("data", "*.txt"))

    for data_file in data_files:
        data_name = os.path.basename(data_file).replace(".txt", "")
        for bs in args.batch_sizes:
            for emb in args.embed_sizes:
                for nl in args.num_layers_list:
                    for lr in args.learning_rates:
                        folder_name = os.path.join(
                            "results",
                            f"{data_name}_bs{bs}_emb{emb}_L{nl}_B{args.block_size}_lr{lr}"
                        )
                        print(f"Fast training: data={data_file}, batch={bs}, embed={emb}, layers={nl}, lr={lr}")
                        # 快速训练
                        try:
                            run_training(
                                data_file, bs, emb, nl, args.block_size,
                                lr, args.fast_epochs, folder_name
                            )
                        except subprocess.CalledProcessError:
                            print(f"Fast training failed for {folder_name}")
                            continue

                        final_loss = read_final_loss(folder_name)
                        if final_loss is None:
                            print(f"No loss info for {folder_name}, skipping full training")
                            continue
                        # 判断是否继续 full training
                        if final_loss <= args.loss_threshold:
                            full_epochs = args.fast_epochs
                            total_epochs = 0
                            last_loss = final_loss
                            while full_epochs <= args.max_epochs:
                                # 逐步增加训练步数
                                try:
                                    run_training(
                                        data_file, bs, emb, nl, args.block_size,
                                        lr, full_epochs, folder_name
                                    )
                                except subprocess.CalledProcessError:
                                    print(f"Full training failed for {folder_name}")
                                    break

                                new_loss = read_final_loss(folder_name)
                                if new_loss is None:
                                    break
                                improvement = last_loss - new_loss
                                if improvement < args.min_improve:
                                    print(f"Loss improvement {improvement:.4f} < min_improve, stop training")
                                    break
                                last_loss = new_loss
                                full_epochs += args.step_epochs
                                total_epochs += args.step_epochs
                            # 训练完成，生成文本
                            model_path = os.path.join(folder_name, 'model.pt')
                            run_generation(model_path, folder_name, args.prompt, args.gen_length)
                            print(f"Completed full training and generation for {folder_name}")
                        else:
                            print(f"Skip full training for {folder_name}, final_loss={final_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adaptive hyperparameter sweep for MiniGPT")
    parser.add_argument('--batch_sizes',   type=int, nargs='+', default=[16,32],    help='List of batch sizes')
    parser.add_argument('--embed_sizes',   type=int, nargs='+', default=[64,128],   help='List of embedding sizes')
    parser.add_argument('--num_layers_list', type=int, nargs='+', default=[2,4],   help='List of number of layers')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[0.001,0.0005], help='List of learning rates')
    parser.add_argument('--block_size',    type=int, default=16,    help='Block (sequence) size')
    parser.add_argument('--fast_epochs',   type=int, default=100,     help='Epochs for fast test training')
    parser.add_argument('--step_epochs',   type=int, default=100,     help='Epochs to add per full training step')
    parser.add_argument('--max_epochs',    type=int, default=5000,    help='Maximum total epochs for full training')
    parser.add_argument('--min_improve',   type=float, default=0.0001, help='Minimum loss improvement to continue training')
    parser.add_argument('--loss_threshold',type=float, default=2.0,  help='Threshold loss to consider continuing full training')
    parser.add_argument('--gen_length',    type=int, default=100,   help='Length of text to generate')
    parser.add_argument('--prompt',        type=str, default='',     help='Optional prompt for generation')
    args = parser.parse_args()
    main(args)
