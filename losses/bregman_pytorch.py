# Code modified from https://github.com/cvlab-stonybrook/DM-Count/blob/master/losses/bregman_pytorch.py
import torch
from torch.amp import autocast
from torch import Tensor
from typing import Union, Tuple, Dict

M_EPS = 1e-16


@torch.no_grad()
@autocast(device_type="cuda", enabled=True, dtype=torch.float32)
def sinkhorn(
    a: Tensor,
    b: Tensor,
    C: Tensor,
    reg: float = 1e-1,
    maxIter: int = 1000,
    stopThr: float = 1e-9,
    verbose: bool = False,
    log: bool = True,
    eval_freq: int = 10,
    print_freq: int = 200,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    device = a.device
    na, nb = C.shape
    assert na == a.shape[0] and nb == b.shape[0], f"Shapes of a ({a.shape}) or b ({b.shape}) do not match that of C ({C.shape})"
    assert reg > 0, f"reg should be greater than 0. Found reg = {reg}"
    assert a.min() >= 0. and b.min() >= 0., f"Elements in a and b should be nonnegative. Found a.min() = {a.min()}, b.min() = {b.min()}"

    if log:
        log = {"err": []}

    u = torch.ones(na, dtype=a.dtype, device=device) / na
    v = torch.ones(nb, dtype=b.dtype, device=device) / nb
    K = torch.exp(-C / reg)

    it, err = 1, 1
    while (err > stopThr and it <= maxIter):
        u_pre, v_pre = u.clone(), v.clone()
        KTu = torch.matmul(K.T, u)
        v = b / (KTu + M_EPS)
        Kv = torch.matmul(K, v)
        u = a / (Kv + M_EPS)

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print("Warning: numerical errors at iteration", it)
            u, v = u_pre, v_pre
            break

        if log and it % eval_freq == 0:
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            log["err"].append(err)

        if verbose and it % print_freq == 0:
            print(f"Iteration {it}, constraint error {err}")

        it += 1

    if log:
        log["u"] = u
        log["v"] = v
        log["alpha"] = reg * torch.log(u + M_EPS)
        log["beta"] = reg * torch.log(v + M_EPS)

    P = u.view(-1, 1) * K * v.view(1, -1)
    if log:
        return P, log
    else:
        return P
