from typing import Tuple, TypedDict, Optional


class Constants(TypedDict):
    """
    Beta values used for the ADAM optimiser
    """

    ADAM_BETAS: Tuple[float, float]

    """
    Epsilon to use for the ADAM optimiser
    """

    ADAM_EPSILON: float

    """
    Approximate number of tokens in a batch
    """

    BATCH_SIZE_TOKENS: int

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
    Label smoothing epsilon
    """

    LABEL_SMOOTHING_EPSILON: float

    """
    The (maximum) number of sentence pairs in the dataset (for quick training)
    """

    MAX_SAMPLES: Optional[int]  # TODO Currently unused

    """
    The number of encoder layers
    """
    N_ENCODER_LAYERS: int

    """
    The approximate number of tokens in a sentence pair batch
    """

    SENTENCE_PAIRS_BATCH_SIZE: int

    """
    Vocabulary size/number of unique tokens
    """
    VOCAB_SIZE: int

    """
    The number of warmup steps in the optimiser scheduler
    """

    WARMUP_STEPS: int

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
    "ADAM_BETAS": (0.9, 0.98),
    "ADAM_EPSILON": 1e-9,
    "BATCH_SIZE_TOKENS": 25_000,
    "DIM_K_Q": 64,
    "DIM_V": 64,
    "HEADS": 8,
    "HIDDEN_DIM": 2_048,
    "LABEL_SMOOTHING_EPSILON": 0.1,
    "MAX_SAMPLES": None,
    "N_ENCODER_LAYERS": 6,
    "OUTPUT_DIM": 512,
    "SENTENCE_PAIRS_BATCH_SIZE": 25_000,
    "VOCAB_SIZE": 37_000,
    "WARMUP_STEPS": 4_000,
}

constants_local: Constants = {
    "ADAM_BETAS": (0.9, 0.98),
    "ADAM_EPSILON": 1e-9,
    "BATCH_SIZE_TOKENS": 100,
    "DIM_K_Q": 64,
    "DIM_V": 64,
    "HEADS": 8,
    "HIDDEN_DIM": 2_048,
    "LABEL_SMOOTHING_EPSILON": 0.1,
    "MAX_SAMPLES": 250,
    "N_ENCODER_LAYERS": 6,
    "OUTPUT_DIM": 512,
    "SENTENCE_PAIRS_BATCH_SIZE": 10,
    "VOCAB_SIZE": 37_000,
    "WARMUP_STEPS": 4_000,
}


misc_constants: MiscConstants = {
    "TOKENIZATION_METHOD": "bpe",
    "SPM_FOLDER": "sentencepiece_model",
    "CORPUS_FOLDER": "corpus",
    "DATASET_TRAIN_URL": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/train.jsonl",
    "DATASET_VALID_URL": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/val.jsonl",
    "DATASET_TEST_URL": "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/test.jsonl",
}
