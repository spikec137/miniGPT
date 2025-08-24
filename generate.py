import argparse
import os
import torch

def generate_text(model_path, output_dir, prompt="", max_length=100):
    # 加载模型和词表
    checkpoint = torch.load(model_path, map_location='cpu')
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    vocab_size = checkpoint['vocab_size']
    embed_size = checkpoint['embed_size']
    num_layers = checkpoint['num_layers']
    block_size = checkpoint['block_size']

    # 定义模型结构并加载权重
    class GPTModelGen(torch.nn.Module):
        def __init__(self, vocab_size, embed_size, num_layers, block_size):
            super().__init__()
            self.token_embedding = torch.nn.Embedding(vocab_size, embed_size)
            self.position_embedding = torch.nn.Embedding(block_size, embed_size)
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_size, nhead=2, batch_first=True)
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.linear = torch.nn.Linear(embed_size, vocab_size)

        def forward(self, x):
            seq_len = x.size(1)
            pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
            tok_emb = self.token_embedding(x)
            pos_emb = self.position_embedding(pos_ids)
            h = tok_emb + pos_emb
            out = self.transformer(h)
            logits = self.linear(out)
            return logits

    model = GPTModelGen(vocab_size, embed_size, num_layers, block_size)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 准备初始输入
    if prompt:
        input_ids = [stoi.get(ch, 0) for ch in prompt]
        current_text = prompt
    else:
        import random
        idx = random.randrange(vocab_size)
        input_ids = [idx]
        current_text = itos[idx]

    # 逐步生成字符
    for _ in range(max_length):
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        logits = model(input_tensor)                # (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :]       # (vocab_size)
        next_id = torch.argmax(next_token_logits).item()
        next_char = itos[next_id]
        current_text += next_char
        input_ids.append(next_id)
        if len(input_ids) > block_size:
            input_ids = input_ids[-block_size:]

    # 保存生成结果
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'generated.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(current_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate text with a trained MiniGPT model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model .pt file')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output text')
    parser.add_argument('--prompt', type=str, default='', help='Optional prompt text to start generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of text to generate')
    args = parser.parse_args()
    generate_text(args.model_path, args.output_dir, prompt=args.prompt, max_length=args.max_length)
