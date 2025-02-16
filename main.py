import os
from constants import constants_local, misc_constants
from model import Encoder, DecoderLayer
from nlp_prep import download_corpora
from data import save_tokenized_model, TextData, Tokenizer
from model import InputEmbedding


def main():
    download_corpora(
        train_url=misc_constants["DATASET_TRAIN_URL"],
        valid_url=misc_constants["DATASET_VALID_URL"],
        test_url=misc_constants["DATASET_TEST_URL"],
        corpus_folder=misc_constants["CORPUS_FOLDER"],
        avoid_overwrite=True,
    )

    save_tokenized_model(
        vocab_size=constants_local["VOCAB_SIZE"],
        corpus_folder=misc_constants["CORPUS_FOLDER"],
        model_type=misc_constants["TOKENIZATION_METHOD"],
        spm_folder=misc_constants["SPM_FOLDER"],
        avoid_overwrite=True,
    )

    train_dataset = TextData(
        dataset_path=os.path.join(misc_constants["CORPUS_FOLDER"], "train.txt"),
    )

    tokenizer = Tokenizer(spm_folder=misc_constants["SPM_FOLDER"])

    x_encoder = tokenizer.encode(train_dataset[0])

    input_embedding = InputEmbedding(
        input_dim=constants_local["OUTPUT_DIM"],
        vocab_size=constants_local["VOCAB_SIZE"],
    )

    x_encoder = input_embedding(x_encoder)

    encoder = Encoder(
        n=constants_local["N_ENCODER_LAYERS"],
        model_dim=constants_local["OUTPUT_DIM"],
        hidden_dim=constants_local["HIDDEN_DIM"],
        w_q_k=constants_local["DIM_K_Q"],
        w_v=constants_local["DIM_V"],
        heads=constants_local["HEADS"],
    )

    x_encoder = encoder(x_encoder)

    decoder_layer = DecoderLayer(
        model_dim=constants_local["OUTPUT_DIM"],
        hidden_dim=constants_local["HIDDEN_DIM"],
        w_q_k=constants_local["DIM_K_Q"],
        w_v=constants_local["DIM_V"],
        heads=constants_local["HEADS"],
    )

    tokenizer.shift_right = True
    x_decoder = tokenizer.encode(train_dataset[0])

    output_embedding = InputEmbedding(
        input_dim=constants_local["OUTPUT_DIM"],
        vocab_size=constants_local["VOCAB_SIZE"],
    )

    x_decoder = output_embedding(x_decoder)

    x_decoder = decoder_layer(x_decoder, x_encoder)

    return


if __name__ == "__main__":
    main()
