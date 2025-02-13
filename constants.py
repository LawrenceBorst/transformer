from typing import TypedDict, Optional


class Constants(TypedDict):
    """
    The (maximum) number of sentence pairs in the dataset (for quick training)
    """

    MAX_SAMPLES: Optional[int]

    """
    The approximate number of tokens in a sentence pair batch
    """

    SENTENCE_PAIRS_BATCH_SIZE: int

    """
    Vocabulary size/number of unique tokens
    """
    VOCAB_SIZE: int


class MiscConstants(TypedDict):
    """
    The local path to save the corpus folder
    """

    CORPUS_FOLDER: str
    """
    The download url for the training dataset
    """
    DATASET_TRAIN_URL: str

    """
    The download url for the validation dataset
    """
    DATASET_VALID_URL: str

    """
    The download url for the test dataset
    """
    DATASET_TEST_URL: str


constants_paper: Constants = {
    "MAX_SAMPLES": None,
    "SENTENCE_PAIRS_BATCH_SIZE": 25_000,
    "VOCAB_SIZE": 37_000,
}

constants_local: Constants = {
    "MAX_SAMPLES": 50,
    "SENTENCE_PAIRS_BATCH_SIZE": 10,
    "VOCAB_SIZE": 37_000,
}


misc_constants: MiscConstants = {
    "CORPUS_FOLDER": "corpus",
    "DATASET_TRAIN_URL": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/train.jsonl",
    "DATASET_VALID_URL": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/val.jsonl",
    "DATASET_TEST_URL": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/test.jsonl",
}
