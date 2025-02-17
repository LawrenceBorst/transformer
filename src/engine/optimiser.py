from typing import Tuple
import torch

from ..model.transformer import Transformer


def get_optimizer(
    model: Transformer,
    betas: Tuple[float, float],
    eps: float,
    d_model: int,
    warmup_steps: int,
    batch_size: int,
    paper_batch_size: int,
) -> Tuple[torch.optim.Adam, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=_learning_rate(
            1,
            d_model,
            warmup_steps,
            batch_size,
            paper_batch_size,
        ),
        betas=betas,
        eps=eps,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: _learning_rate(
            step=step + 1,
            d_model=d_model,
            warmup_steps=warmup_steps,
            batch_size=batch_size,
            paper_batch_size=paper_batch_size,
        ),
    )

    return optimizer, scheduler


def _learning_rate(
    step: int, d_model: int, warmup_steps: int, batch_size: int, paper_batch_size: int
) -> float:
    adjusted_step: int = batch_size / paper_batch_size * step

    return (d_model**0.5) * min(adjusted_step**-0.5, adjusted_step * warmup_steps**-1.5)
