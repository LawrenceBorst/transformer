import torch
from .encoder_layer import EncoderLayer


class Encoder(torch.nn.Module):
    """
    A stack of encoder layers

    Args:
        n (int): the number of encoder layers
        model_dim (int): the model dimension
        hidden_dim (int): the hidden dimension in the feedforward layer
        w_q_k (int): the 1st dimension of the query and key matrices
        w_v (int): the 1st dimension of the value matrix
        heads (int): number of attention
    """

    _encoder_layers: torch.nn.ModuleList

    def __init__(
        self, n: int, model_dim: int, hidden_dim: int, w_q_k: int, w_v: int, heads: int
    ) -> None:
        super().__init__()

        self._encoder_layers = torch.nn.ModuleList(
            [
                EncoderLayer(
                    model_dim=model_dim,
                    hidden_dim=hidden_dim,
                    w_q_k=w_q_k,
                    w_v=w_v,
                    heads=heads,
                )
                for _ in range(n)
            ]
        )

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for el in self._encoder_layers:
            x = el(x)

        return x
