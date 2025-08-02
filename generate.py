# generate.py
import torch
from model import MiniGPT

ckpt = torch.load('checkpoints/mini-gpt.pt')
model = MiniGPT(vocab_size=len(ckpt['stoi']), block_size=ckpt['block_size'])
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

stoi = ckpt['stoi']
itos = ckpt['itos']

context = torch.tensor([[stoi['你']]], dtype=torch.long)
print(model.generate(context, max_new_tokens=100, stoi=stoi, itos=itos))
