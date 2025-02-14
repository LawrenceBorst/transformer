from torch.utils.data import Dataset


class TextData(Dataset[str]):
    _data: list[str]

    def __init__(self, dataset_path: str) -> None:
        self._data = self._get_data(dataset_path)

    def _get_data(self, dataset_path: str) -> list[str]:
        with open(dataset_path, "r") as f:
            return [line.rstrip("\n") for line in f]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> str:
        return self._data[idx]
