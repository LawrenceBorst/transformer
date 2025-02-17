import torch


def get_loss_function() -> torch.nn.CrossEntropyLoss:
    return torch.nn.CrossEntropyLoss()
