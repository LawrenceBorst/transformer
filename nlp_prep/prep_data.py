import os
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
    downloaders: list[CorpusDownloader] = [
        CorpusDownloader(
            dataset_url=train_url,
            output_path=os.path.join(corpus_folder, "train.txt"),
            avoid_overwrite=avoid_overwrite,
        ),
        CorpusDownloader(
            dataset_url=valid_url,
            output_path=os.path.join(corpus_folder, "valid.txt"),
            avoid_overwrite=avoid_overwrite,
        ),
        CorpusDownloader(
            dataset_url=test_url,
            output_path=os.path.join(corpus_folder, "test.txt"),
            avoid_overwrite=avoid_overwrite,
        ),
    ]

    for downloader in downloaders:
        downloader.save()
