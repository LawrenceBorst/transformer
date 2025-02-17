import os
from typing import Tuple
from torch.utils.data.datapipes.iter.sharding import ShardingFilterIterDataPipe
import shutil
import sentencepiece as spm


def save_tokenized_model(
    vocab_size: int,
    corpus_folder: str,
    spm_folder: str,
    model_type: str = "bpe",
    avoid_overwrite: bool = False,
) -> Tuple[
    ShardingFilterIterDataPipe,
    ShardingFilterIterDataPipe,
    ShardingFilterIterDataPipe,
]:
    if (
        avoid_overwrite
        and os.path.exists(f"{spm_folder}/spm.model")
        and os.path.exists(f"{spm_folder}/spm.vocab")
    ):
        return

    os.makedirs(spm_folder, exist_ok=True)

    train_path, valid_path, test_path = _get_corpus_paths(corpus_folder)

    spm.SentencePieceTrainer.train(
        input=f"{train_path},{valid_path},{test_path}",
        model_prefix="spm",
        model_type=model_type,
        vocab_size=vocab_size,
        accept_language="en,de",
    )

    _move_to_sentencepiece_folder(spm_folder)


def _get_corpus_paths(
    corpus_folder: str,
) -> Tuple[
    str, str, str
]:  # TODO: Duplicated function, but need to do some rearranging to allow importing it
    return (
        os.path.join(corpus_folder, "train.txt"),
        os.path.join(corpus_folder, "valid.txt"),
        os.path.join(corpus_folder, "test.txt"),
    )


def _move_to_sentencepiece_folder(spm_folder: str) -> None:
    """
    This is dumb, but the spm API does not let us specify an output path, so we do this
    """
    shutil.move("spm.model", f"{spm_folder}/spm.model")
    shutil.move("spm.vocab", f"{spm_folder}/spm.vocab")
