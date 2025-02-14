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

    def __init__(self, spm_folder: str) -> None:
        self._sp = spm.SentencePieceProcessor(
            model_file=os.path.join(spm_folder, "spm.model")
        )

    def encode(self, sentence: str) -> torch.IntTensor:
        return torch.IntTensor(self._sp.encode(sentence))
    
    def decode(self, sentence: torch.IntTensor) -> str:
        return self._sp.decode(sentence.tolist())
