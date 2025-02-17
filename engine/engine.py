import math
from typing import List, Tuple
import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm


class Engine:
    """
    Engine for training and evaluating a model

    Args:
        model (torch.nn.Module): The model to train
        train_loader (torch.utils.data.DataLoader[str]): The training data loader
        valid_loader (torch.utils.data.DataLoader[str]): the validation data loader
        test_loader (torch.utils.data.DataLoader[str]): the test data loader
        scheduler (torch.optim.lr_scheduler): the optimiser scheduler
        optimizer (torch.optim.Optimizer): the optimizer to use
        loss_fn (torch.nn.Module): the loss function to use
        epochs (int): the number of epochs to train for
        device (torch.device): the device to train on
        tokenizer (Tokenizer): the tokenizer to use
    """

    _model: torch.nn.Module
    _train_loader: torch.utils.data.DataLoader[str]
    _valid_loader: torch.utils.data.DataLoader[str]
    _test_loader: torch.utils.data.DataLoader[str]
    _optimizer: torch.optim.Optimizer
    _scheduler: torch.optim.lr_scheduler
    _loss_fn: torch.nn.CrossEntropyLoss
    _epochs: int
    _device: torch.device

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader[str],
        valid_loader: torch.utils.data.DataLoader[str],
        test_loader: torch.utils.data.DataLoader[str],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: torch.nn.CrossEntropyLoss,
        epochs: int,
        device: torch.device,
        tokenizer,
    ) -> None:
        self._model = model
        self._train_loader = train_loader
        self._valid_loader = valid_loader
        self._test_loader = test_loader
        self._scheduler = scheduler
        self._optimizer = optimizer
        self._loss_fn = loss_fn
        self._epochs = epochs
        self._device = device
        self._tokenizer = tokenizer

    def validate(self) -> float:
        self._model.eval()

        total_loss: float = 0

        with torch.inference_mode():
            for idx, y in enumerate(self._valid_loader):
                tokens_in: torch.IntTensor = self._tokenizer.encode(y).to(self._device)
                tokens_out: torch.IntTensor = self._tokenizer.encode(
                    y, shift_right=True
                ).to(self._device)

                token_preds = self._model(tokens_in, tokens_out)

                loss: torch.Tensor = self._loss_fn(
                    input=token_preds[1:], target=tokens_out.long()[1:]
                )

                total_loss += loss.item()

        return total_loss

    def train(self) -> List[Tuple[float, List[float]]]:
        self._model.train()

        losses: List[Tuple[float, List[float]]] = []

        for _ in range(self._epochs):
            epoch_losses = self._train_epoch()

            losses.append(epoch_losses)

        return losses

    def _train_epoch(self) -> Tuple[float, List[float]]:
        total_loss: float = 0
        cumulative_loss: List[float] = []
        last_reported_loss: float = 0

        for idx, y in tqdm(enumerate(self._train_loader)):
            tokens_in: torch.IntTensor = self._tokenizer.encode(y).to(self._device)
            tokens_out: torch.IntTensor = self._tokenizer.encode(
                y, shift_right=True
            ).to(self._device)

            token_preds = self._model(tokens_in, tokens_out)

            loss: torch.Tensor = self._loss_fn(
                input=token_preds[1:], target=tokens_out.long()[1:]
            )

            total_loss += loss.item()

            self._optimizer.zero_grad()

            loss.backward()

            self._optimizer.step()
            self._scheduler.step()

            if idx % 25 == 0:
                print(f"loss: {total_loss}")
                print(f"delta loss: {total_loss - last_reported_loss}")

                last_reported_loss = total_loss
                cumulative_loss.append(total_loss)

        return total_loss, cumulative_loss
