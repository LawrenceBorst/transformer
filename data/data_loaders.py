import os
from typing import Optional, Tuple
from torch.utils.data.dataloader import DataLoader

from data.dataset import TextData


def get_data_loaders(
    corpus_folder: str, max_samples: Optional[int]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = TextData(
        dataset_path=os.path.join(corpus_folder, "train.txt"),
        max_samples=max_samples,
    )
    valid_dataset = TextData(
        dataset_path=os.path.join(corpus_folder, "valid.txt"),
        max_samples=max_samples,
    )
    test_dataset = TextData(
        dataset_path=os.path.join(corpus_folder, "test.txt"),
        max_samples=max_samples,
    )

    train_loader = DataLoader(train_dataset, batch_size=None)
    valid_loader = DataLoader(valid_dataset, batch_size=None)
    test_loader = DataLoader(test_dataset, batch_size=None)

    return train_loader, valid_loader, test_loader
