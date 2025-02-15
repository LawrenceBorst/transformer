import torch
from .feed_forward import FeedForward
from .attention import Attention


class EncoderLayer(torch.nn.Module):
    """
    A single encoder layer. This combines multihead attention, a skip connection and layer norm, feedforward layer, and another skip connection and layer norm

    Args:
        model_dim (int): the model dimension
        hidden_dim (int): the hidden dimension in the feedforward layer
        w_q_k (int): the 1st dimension of the query and key matrices
        w_v (int): the 1st dimension of the value matrix
        heads (int): number of attention
    """

    _attention: Attention
    _feedforward: FeedForward
    _ln_1: torch.nn.LayerNorm
    _ln_2: torch.nn.LayerNorm

    def __init__(
        self, model_dim: int, hidden_dim: int, w_q_k: int, w_v: int, heads: int
    ) -> None:
        super().__init__()

        self._attention = Attention(
            model_dim=model_dim, w_q_k=w_q_k, w_v=w_v, heads=heads
        )
        self._feedforward = FeedForward(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
        )
        # No mention of learnable layer norm in the paper
        self._ln_1 = torch.nn.LayerNorm(
            normalized_shape=model_dim, elementwise_affine=False
        )
        self._ln_2 = torch.nn.LayerNorm(
            normalized_shape=model_dim, elementwise_affine=False
        )

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ln_1(x + self._attention(x))
        x = self._ln_2(x + self._feedforward(x))

        return x
