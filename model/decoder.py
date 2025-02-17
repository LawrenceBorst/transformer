import torch
from .decoder_layer import DecoderLayer


class Decoder(torch.nn.Module):
    """
    A stack of decoder layers

    Args:
        n (int): the number of decoder layers
        model_dim (int): the model dimension
        hidden_dim (int): the hidden dimension in the feedforward layer
        w_q_k (int): the 1st dimension of the query and key matrices
        w_v (int): the 1st dimension of the value matrix
        heads (int): number of attention
        device (torch.device): the torch device
    """

    _decoder_layers: torch.nn.ModuleList

    def __init__(
        self,
        n: int,
        model_dim: int,
        hidden_dim: int,
        w_q_k: int,
        w_v: int,
        heads: int,
        device: torch.device,
    ) -> None:
        super().__init__()

        self._decoder_layers = torch.nn.ModuleList(
            [
                DecoderLayer(
                    model_dim=model_dim,
                    hidden_dim=hidden_dim,
                    w_q_k=w_q_k,
                    w_v=w_v,
                    heads=heads,
                    device=device,
                )
                for _ in range(n)
            ]
        )

        return

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        for el in self._decoder_layers:
            x = el(
                x,
                encoder_output,
            )

        return x
