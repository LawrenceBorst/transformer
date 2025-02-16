import os
import sentencepiece as spm
import torch


class Tokenizer:
    """
    Handles miscellaneous tokenization tasks using SentencePiece.

    Args:
        spm_folder (str): the folder to the spm model
    """

    _sp: spm.SentencePieceProcessor
    _shift_right: bool

    def __init__(self, spm_folder: str) -> None:
        self._sp = spm.SentencePieceProcessor(
            model_file=os.path.join(spm_folder, "spm.model")
        )
        self._shift_right = False

    @property
    def shift_right(self) -> bool:
        return self._shift_right

    @shift_right.setter
    def shift_right(self, val: bool) -> None:
        self._shift_right = val

    def encode(self, sentence: str) -> torch.IntTensor:
        result: torch.IntTensor = torch.IntTensor(self._sp.encode(sentence))

        if self._shift_right:
            result = torch.cat([torch.IntTensor([1]), result])

        return result

    def decode(self, sentence: torch.IntTensor) -> str:
        return self._sp.decode(sentence.tolist())
