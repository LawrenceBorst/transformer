import os
from typing import Tuple
from .corpus_downloader import CorpusDownloader


def download_corpora(
    train_url: str,
    valid_url: str,
    test_url: str,
    corpus_folder: str,
    avoid_overwrite: bool = False,
):
    """
    Downloads all the corpora (train, validation, and test) from the provided URLs and saves them to the respective files. If `avoid_overwrite` is set to True, it will avoid overwriting existing files.
    """
    train_path, valid_path, test_path = get_corpus_paths(corpus_folder)

    downloaders: list[CorpusDownloader] = [
        CorpusDownloader(
            dataset_url=train_url,
            output_path=train_path,
            avoid_overwrite=avoid_overwrite,
        ),
        CorpusDownloader(
            dataset_url=valid_url,
            output_path=valid_path,
            avoid_overwrite=avoid_overwrite,
        ),
        CorpusDownloader(
            dataset_url=test_url,
            output_path=test_path,
            avoid_overwrite=avoid_overwrite,
        ),
    ]

    for downloader in downloaders:
        downloader.save()


def get_corpus_paths(corpus_folder: str) -> Tuple[str, str, str]:
    return (
        os.path.join(corpus_folder, "train.txt"),
        os.path.join(corpus_folder, "valid.txt"),
        os.path.join(corpus_folder, "test.txt"),
    )
