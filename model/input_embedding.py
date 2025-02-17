import math
import torch


class InputEmbedding(torch.nn.Module):
    """
    This class provides input embeddings for a transformer model. It combines token embeddings with positional encodings to retain the order of tokens.

    Args:
        input_dim (int): the size of the input
        vocab_size (int): the size of the vocabulary
        dropout (float): dropout probability
        device (torch.device): the torch device
    """

    _input_embedding: torch.nn.Embedding
    _model_dim: int
    _dropout: torch.nn.Dropout
    _device: torch.device

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        dropout: float,
        device: torch.device,
    ) -> None:
        super().__init__()

        self._input_embedding = torch.nn.Embedding(
            embedding_dim=input_dim,
            num_embeddings=vocab_size,
            device=device,
        )
        self._model_dim = input_dim
        self._dropout = torch.nn.Dropout(p=dropout)
        self._device = device

        return

    def _get_pos_encoding(self, x_length: int) -> torch.Tensor:
        """
        Returns the positional encodings, as a tensor of shape [x_length, self._model_dim]

        Here the first dimension runs over the position in the embedded token, and the second over the model dimension
        So that a single row corresponds to a positional encoding for a single token embedding
        """
        pe = torch.zeros(x_length, self._model_dim)

        positions: torch.Tensor = torch.arange(
            0, x_length, dtype=torch.float
        ).unsqueeze(
            -1
        )  # Shape [x_length, 1]

        trig_args: torch.Tensor = torch.exp(
            torch.arange(0, self._model_dim, 2).float()
            * -(math.log(10_000.0) / self._model_dim)
        )  # Shape [self._model_dim / 2]

        # Weave in sin and cosine terms
        pe[:, 0::2] = torch.sin(positions * trig_args)
        pe[:, 1::2] = torch.cos(positions * trig_args)

        return pe.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_encoding: torch.Tensor = self._get_pos_encoding(len(x))

        return self._dropout(
            self._input_embedding(x) * math.sqrt(self._model_dim) + pos_encoding
        )
