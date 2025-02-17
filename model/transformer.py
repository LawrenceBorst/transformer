import torch

from .input_embedding import InputEmbedding
from .encoder import Encoder
from .decoder import Decoder


class Transformer(torch.nn.Module):
    """
    A transformer module

    Args:
        output_dim (int): The dimension of the output
        vocab_size (int): The size of the vocabulary
        n_encoder_layers (int): The number of encoder/decoder layers
        hidden_dim (int): The dimension of the hidden layer in the feed forward layers
        w_q_k (int): The dimension of the query and key matrices
        w_v (int): The dimension of the value matrices
        heads (int): The number of attention heads
        device (torch.device): the torch device
    """

    _input_embedding: InputEmbedding
    _output_embedding: InputEmbedding

    _encoder: Encoder
    _decoder: Decoder

    def __init__(
        self,
        output_dim: int,
        vocab_size: int,
        n_encoder_layers: int,
        hidden_dim: int,
        w_q_k: int,
        w_v: int,
        heads: int,
        device: torch.device,
    ) -> None:
        super().__init__()

        self._input_embedding = InputEmbedding(
            input_dim=output_dim,
            vocab_size=vocab_size,
            device=device,
        )
        self._output_embedding = InputEmbedding(
            input_dim=output_dim,
            vocab_size=vocab_size,
            device=device,
        )

        self._encoder = Encoder(
            n=n_encoder_layers,
            model_dim=output_dim,
            hidden_dim=hidden_dim,
            w_q_k=w_q_k,
            w_v=w_v,
            heads=heads,
            device=device,
        )

        self._decoder = Decoder(
            n=n_encoder_layers,
            model_dim=output_dim,
            hidden_dim=hidden_dim,
            w_q_k=w_q_k,
            w_v=w_v,
            heads=heads,
            device=device,
        )

        self._linear = torch.nn.Linear(
            output_dim,
            vocab_size,
            device=device,
        )

        return

    def forward(self, x_input: torch.Tensor, x_output) -> torch.Tensor:
        x_input = self._input_embedding(x_input)

        x_input = self._encoder(x_input)

        x_output = self._output_embedding(x_output)

        x_output = self._decoder(x_output, x_input)

        x_output = self._linear(x_output)

        return x_output
