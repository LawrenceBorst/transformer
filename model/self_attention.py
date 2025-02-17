from .attention import Attention
import torch


class SelfAttention(torch.nn.Module, Attention):
    """
    A trainable multi-head self-attention module

    Args:
        model_dim (int): the input and output dimension
        w_q_k (int): the query/key dimension
        w_v (int): the value dimension
        heads (int): number of heads
        device (torch.device): the torch device
        masked (bool): if using masked attention. Defaults to false
    """

    def __init__(
        self,
        model_dim: int,
        w_q_k: int,
        w_v: int,
        heads: int,
        device: torch.device,
        masked: bool = False,
    ) -> None:
        torch.nn.Module.__init__(self)
        Attention.__init__(
            self,
            model_dim=model_dim,
            w_q_k=w_q_k,
            w_v=w_v,
            heads=heads,
            device=device,
            masked=masked,
        )

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = [q(x) for q in self._query_matrices]
        k = [k(x) for k in self._key_matrices]
        v = [v(x) for v in self._value_matrices]

        multihead: torch.Tensor = torch.concat(
            [self._attention(q[h], k[h], v[h]) for h in range(self._heads)], dim=-1
        )

        return self._output(multihead)
