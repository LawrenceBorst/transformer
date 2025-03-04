import torch
from .feed_forward import FeedForward
from .self_attention import SelfAttention


class EncoderLayer(torch.nn.Module):
    """
    A single encoder layer. This combines multihead attention, a skip connection and
    layer norm, feedforward layer, and another skip connection and layer norm

    Args:
        model_dim (int): the model dimension
        hidden_dim (int): the hidden dimension in the feedforward layer
        w_q_k (int): the 1st dimension of the query and key matrices
        w_v (int): the 1st dimension of the value matrix
        heads (int): number of attention
        dropout (float): dropout probability
        device (torch.device): the torch device
    """

    _attention: SelfAttention
    _feedforward: FeedForward
    _ln_1: torch.nn.LayerNorm
    _ln_2: torch.nn.LayerNorm
    _dropout: torch.nn.Dropout

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        w_q_k: int,
        w_v: int,
        heads: int,
        dropout: float,
        device: torch.device,
    ) -> None:
        super().__init__()

        self._attention = SelfAttention(
            model_dim=model_dim,
            w_q_k=w_q_k,
            w_v=w_v,
            heads=heads,
            device=device,
        )
        self._feedforward = FeedForward(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            device=device,
        )
        # No mention of learnable layer norm in the paper
        self._ln_1 = torch.nn.LayerNorm(
            normalized_shape=model_dim,
            elementwise_affine=False,
            device=device,
        )
        self._ln_2 = torch.nn.LayerNorm(
            normalized_shape=model_dim,
            elementwise_affine=False,
            device=device,
        )
        self._dropout = torch.nn.Dropout(
            p=dropout,
        )

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ln_1(x + self._attention(self._dropout(x)))
        x = self._ln_2(x + self._feedforward(self._dropout(x)))

        return x
