import torch
import math

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
