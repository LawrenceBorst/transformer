import torch


def get_loss_function(label_smoothing: float) -> torch.nn.CrossEntropyLoss:
    return torch.nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
    )
