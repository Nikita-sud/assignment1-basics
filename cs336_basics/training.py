import torch
import torch.nn.functional as F
import math
import numpy as np
import os
import typing
from typing import Optional, Callable
import time

from cs336_basics.model import TransformerLM

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    m = torch.max(logits, dim=-1, keepdim=True).values
    
    stable_logits = logits - m

    lse = torch.log(torch.sum(torch.exp(stable_logits), dim=-1))

    target_logits = torch.gather(stable_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    loss = lse - target_logits

    return torch.mean(loss)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_size = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)
                
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        
        return loss
    
def learning_rate_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int):
    if t < T_w:
        return alpha_max * t / T_w
    elif T_w <= t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * progress))
    else:
        return alpha_min
    
def gradient_clipping(model, max_norm):
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0

    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 
        2
    )

    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for p in params:
            p.grad.detach().mul_(clip_coef)
            
    return total_norm

def get_batch(data: np.ndarray, batch_size: int, context_length: int, device: str):

    max_idx = len(data) - context_length - 1
    ix = torch.randint(0, max_idx, (batch_size,))
    
    x_chunks = [torch.from_numpy(data[i : i + context_length].astype(np.int64)) for i in ix]
    y_chunks = [torch.from_numpy(data[i + 1 : i + context_length + 1].astype(np.int64)) for i in ix]
    
    x = torch.stack(x_chunks)
    y = torch.stack(y_chunks)
    
    return x.to(device), y.to(device)


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint["model"])
    
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["iteration"]



def train():
    vocab_size = 10000
    context_length = 256
    d_model = 512
    d_ff = 1344
    num_layers = 4
    num_heads = 16
    
    batch_size = 32
    max_steps = 5000
    
    alpha_max = 5e-4
    alpha_min = 1e-5
    T_w = 100
    T_c = max_steps
    max_norm = 1.0
    
    eval_interval = 250
    save_interval = 1000
    out_dir = "checkpoints"
    os.makedirs(out_dir, exist_ok=True)

    device = 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device
    )
    if device != 'mps':
        model = torch.compile(model)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=alpha_max, weight_decay=0.01, betas=(0.9, 0.95))

    train_data = np.memmap('data/tinystories_train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('data/tinystories_val.bin', dtype=np.uint16, mode='r')

    token_positions = torch.arange(context_length, device=device).unsqueeze(0).expand(batch_size, -1)
    offsets = np.arange(context_length)

    def fast_get_batch(data, batch_size, context_length, device):
        starts = np.random.randint(0, len(data) - context_length, size=batch_size)
        idx = starts[:, None] + offsets[None, :]
        x = torch.from_numpy(data[idx].astype(np.int64)).to(device)
        y = torch.from_numpy(data[idx + 1].astype(np.int64)).to(device)
        return x, y

    from tqdm import tqdm

    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:

        lr = learning_rate_schedule(step, alpha_max, alpha_min, T_w, T_c)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = fast_get_batch(val_data, batch_size, context_length, device)
                val_logits = model(x_val, token_positions)
                val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), y_val.view(-1))
            model.train()
            pbar.set_postfix(val_loss=f"{val_loss.item():.4f}", lr=f"{lr:.6f}")

        x, y = fast_get_batch(train_data, batch_size, context_length, device)

        logits = model(x, token_positions)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        if step > 0 and step % save_interval == 0:
            checkpoint_path = os.path.join(out_dir, f"ckpt_step_{step}.pt")
            save_checkpoint(model, optimizer, step, checkpoint_path)
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete!")

if __name__ == "__main__":
    train()
    pass