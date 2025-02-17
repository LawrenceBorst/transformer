import torch
import os
from pathlib import Path


def save_model(model: torch.nn.Module, target_dir: str) -> None:
    target_dir_path: Path = Path(target_dir)
    target_dir_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        obj=model.state_dict(),
        f=os.path.join(target_dir, "model.pth"),
    )
