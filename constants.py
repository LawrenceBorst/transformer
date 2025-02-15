from typing import TypedDict, Optional


class Constants(TypedDict):
    """
    The 1st dimension of the key and query matrices
    """

    DIM_K_Q: int

    """
    The 1st dimension of the value matrices
    """

    DIM_V: int

    """
    The number of attention heads
    """

    HEADS: int

    """
    The hidden dimension in the feedforward modules
    """

    HIDDEN_DIM: int

    """
    The (maximum) number of sentence pairs in the dataset (for quick training)
    """

    MAX_SAMPLES: Optional[int]  # TODO Currently unused

    """
    The approximate number of tokens in a sentence pair batch
    """

    SENTENCE_PAIRS_BATCH_SIZE: int

    """
    Vocabulary size/number of unique tokens
    """
    VOCAB_SIZE: int

    """
    The output dimension of all sublayers, as well as the embedding layer
    """

    OUTPUT_DIM: int


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

    """
    The tokenization algorithm to use
    """
    TOKENIZATION_METHOD: str

    """
    The model folder for the sentencepiece model
    """
    SPM_FOLDER: str


constants_paper: Constants = {
    "DIM_K_Q": 64,
    "DIM_V": 64,
    "HEADS": 8,
    "HIDDEN_DIM": 2048,
    "MAX_SAMPLES": None,
    "OUTPUT_DIM": 512,
    "SENTENCE_PAIRS_BATCH_SIZE": 25_000,
    "VOCAB_SIZE": 37_000,
}

constants_local: Constants = {
    "DIM_K_Q": 64,
    "DIM_V": 64,
    "HEADS": 8,
    "HIDDEN_DIM": 2048,
    "MAX_SAMPLES": 50,
    "OUTPUT_DIM": 512,
    "SENTENCE_PAIRS_BATCH_SIZE": 10,
    "VOCAB_SIZE": 37_000,
}


misc_constants: MiscConstants = {
    "TOKENIZATION_METHOD": "bpe",
    "SPM_FOLDER": "sentencepiece_model",
    "CORPUS_FOLDER": "corpus",
    "DATASET_TRAIN_URL": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/train.jsonl",
    "DATASET_VALID_URL": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/val.jsonl",
    "DATASET_TEST_URL": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/test.jsonl",
}
