from .attention import Attention
import torch


class CrossAttention(torch.nn.Module, Attention):
    """
    A trainable multi-head cross-attention module

    Args:
        model_dim (int): the input and output dimension
        w_q_k (int): the query/key dimension
        w_v (int): the value dimension
        heads (int): number of heads
        device (torch.device): the torch device
    """

    def __init__(
        self,
        model_dim: int,
        w_q_k: int,
        w_v: int,
        heads: int,
        device: torch.device,
    ) -> None:
        torch.nn.Module.__init__(self)
        Attention.__init__(
            self,
            model_dim=model_dim,
            w_q_k=w_q_k,
            w_v=w_v,
            heads=heads,
            masked=False,
            device=device,
        )

        return

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        q = [q(x) for q in self._query_matrices]
        k = [k(encoder_output) for k in self._key_matrices]
        v = [v(encoder_output) for v in self._value_matrices]

        multihead: torch.Tensor = torch.concat(
            [self._attention(q[h], k[h], v[h]) for h in range(self._heads)],
            dim=-1,
        )

        return self._output(multihead)
