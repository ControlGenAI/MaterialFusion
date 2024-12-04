import torch


def use_deterministic():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def toggle_grad(model, mode=True):
    for p in model.parameters():
        p.requires_grad = mode
