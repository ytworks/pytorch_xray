import torch


def clipping(model, is_clip, is_norm, value):
    if is_clip:
        if is_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), value)
        else:
            torch.nn.utils.clip_grad_value_(model.parameters(), value)
