import math
import torch


class SelfAttention(torch.nn.Module):
    """
    A trainable multi-head self-attention module

    # TODO Vectorise the list comprehensions

    Args:
        model_dim (int): the input and output dimension
        w_q_k (int): the query/key dimension
        w_v (int): the value dimension
        heads (int): number of heads
        masked (bool): if using masked attention. Defaults to false
    """

    _query_matrices: list[torch.nn.Linear]
    _key_matrices: list[torch.nn.Linear]
    _value_matrices: list[torch.nn.Linear]
    _w_q_k: int
    _heads: int
    _masked: bool

    def __init__(
        self, model_dim: int, w_q_k: int, w_v: int, heads: int, masked: bool = False
    ) -> None:
        super().__init__()

        # Bias not used in the paper
        self._query_matrices = [
            torch.nn.Linear(model_dim, w_q_k, bias=False) for _ in range(heads)
        ]
        self._key_matrices = [
            torch.nn.Linear(model_dim, w_q_k, bias=False) for _ in range(heads)
        ]
        self._value_matrices = [
            torch.nn.Linear(model_dim, w_v, bias=False) for _ in range(heads)
        ]
        self._output = torch.nn.Linear(w_v * heads, model_dim, bias=False)

        self._w_q_k = w_q_k
        self._heads = heads
        self._masked = masked

        # We opt for He initialization, as the linear layers are followed by a ReLU non-linearity as per the paper
        for h in range(heads):
            torch.nn.init.kaiming_normal_(
                self._query_matrices[h].weight, nonlinearity="relu"
            )
            torch.nn.init.kaiming_normal_(
                self._key_matrices[h].weight, nonlinearity="relu"
            )
            torch.nn.init.kaiming_normal_(
                self._value_matrices[h].weight, nonlinearity="relu"
            )

        torch.nn.init.kaiming_normal_(self._output.weight, nonlinearity="relu")

        return

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scores: torch.Tensor = q @ k.transpose(-2, -1) / math.sqrt(self._w_q_k)

        if self._masked:
            mask_dim: int = q.shape[0]
            mask: torch.tensor = torch.tril(torch.ones(mask_dim, mask_dim))
            scores = scores.masked_fill(mask == 0, float("-inf"))

        return torch.softmax(scores, dim=-1) @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = [q(x) for q in self._query_matrices]
        k = [k(x) for k in self._key_matrices]
        v = [v(x) for v in self._value_matrices]

        multihead: torch.Tensor = torch.concat(
            [self._attention(q[h], k[h], v[h]) for h in range(self._heads)], dim=-1
        )

        return self._output(multihead)
