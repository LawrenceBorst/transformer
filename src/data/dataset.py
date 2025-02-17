import sys
from typing import Optional
from torch.utils.data import Dataset


class TextData(Dataset[str]):
    """
    A class for getting text data

    Args:
        dataset_path (str): the path to the dataset
        max_samples (int): the maximum number of samples, for limiting the size of the dataset
    """

    _data: list[str]
    _max_samples: int

    def __init__(self, dataset_path: str, max_samples: Optional[int]) -> None:
        if max_samples is None:
            max_samples = sys.maxsize

        self._data = self._get_data(dataset_path)
        self._max_samples = max_samples

    def _get_data(self, dataset_path: str) -> list[str]:
        with open(dataset_path, "r") as f:
            return [line.rstrip("\n") for line in f]

    def __len__(self) -> int:
        return min(self._max_samples, len(self._data))

    def __getitem__(self, idx: int) -> str:
        return self._data[idx]
