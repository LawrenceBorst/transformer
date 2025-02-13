from constants import constants_local, misc_constants
from nlp_prep import download_corpora


def main():
    download_corpora(
        train_url=misc_constants["DATASET_TRAIN_URL"],
        valid_url=misc_constants["DATASET_VALID_URL"],
        test_url=misc_constants["DATASET_TEST_URL"],
        corpus_folder=misc_constants["CORPUS_FOLDER"],
        avoid_overwrite=True,
    )

    return


if __name__ == "__main__":
    main()
