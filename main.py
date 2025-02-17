from constants import constants_local, misc_constants, constants_paper
from data.tokenizer import Tokenizer
from nlp_prep import download_corpora
from data import save_tokenized_model, get_data_loaders
from model import Transformer
from engine import Engine, get_optimizer, get_loss_function, set_seed, get_device
import torch


def main():
    device: torch.device = get_device()
    set_seed()

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

    transformer = Transformer(
        output_dim=constants_local["OUTPUT_DIM"],
        vocab_size=constants_local["VOCAB_SIZE"],
        n_encoder_layers=constants_local["N_ENCODER_LAYERS"],
        hidden_dim=constants_local["HIDDEN_DIM"],
        w_q_k=constants_local["DIM_K_Q"],
        w_v=constants_local["DIM_V"],
        heads=constants_local["HEADS"],
        device=device,
    )

    optimizer, scheduler = get_optimizer(
        model=transformer,
        betas=constants_local["ADAM_BETAS"],
        eps=constants_local["ADAM_EPSILON"],
        d_model=constants_local["OUTPUT_DIM"],
        warmup_steps=constants_local["WARMUP_STEPS"],
        batch_size=constants_local["BATCH_SIZE_TOKENS"],
        paper_batch_size=constants_paper["BATCH_SIZE_TOKENS"],
    )

    loss_fn = get_loss_function(
        label_smoothing=constants_local["LABEL_SMOOTHING_EPSILON"],
    )

    train_loader, valid_loader, test_loader = get_data_loaders(
        misc_constants["CORPUS_FOLDER"],
        max_samples=constants_local["MAX_SAMPLES"],
    )

    engine = Engine(
        model=transformer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=1,
        device=device,
        tokenizer=Tokenizer(
            spm_folder=misc_constants["SPM_FOLDER"],
        ),
    )

    losses = engine.train()
    print(losses)
    valid_loss = engine.validate()
    print(valid_loss)

    return


if __name__ == "__main__":
    main()
