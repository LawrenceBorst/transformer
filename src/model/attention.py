import math
import torch


class Attention:
    """
    Base class for cross and self attention

    # TODO Vectorise the list comprehensions

    Args:
        model_dim (int): The dimension of the model
        w_q_k (int): The dimension of the query and key matrices
        w_v (int): The dimension of the value matrices
        heads (int): The number of attention heads
        device (torch.device): the torch device
        masked (bool): Whether to apply masking
    """

    _query_matrices: list[torch.nn.Linear]
    _key_matrices: list[torch.nn.Linear]
    _value_matrices: list[torch.nn.Linear]
    _w_q_k: int
    _heads: int
    _masked: bool
    _device: torch.device

    def __init__(
        self,
        model_dim: int,
        w_q_k: int,
        w_v: int,
        heads: int,
        device: torch.device,
        masked: bool = False,
    ) -> None:
        # Bias not used in the paper
        self._query_matrices = [
            torch.nn.Linear(
                model_dim,
                w_q_k,
                bias=False,
                device=device,
            )
            for _ in range(heads)
        ]
        self._key_matrices = [
            torch.nn.Linear(
                model_dim,
                w_q_k,
                bias=False,
                device=device,
            )
            for _ in range(heads)
        ]
        self._value_matrices = [
            torch.nn.Linear(
                model_dim,
                w_v,
                bias=False,
                device=device,
            )
            for _ in range(heads)
        ]
        self._output = torch.nn.Linear(
            w_v * heads,
            model_dim,
            bias=False,
            device=device,
        )

        self._w_q_k = w_q_k
        self._heads = heads
        self._masked = masked
        self._device = device

        # We opt for He initialization, as the linear layers are followed by a ReLU
        # non-linearity as per the paper
        for h in range(heads):
            torch.nn.init.kaiming_normal_(
                self._query_matrices[h].weight,
                nonlinearity="relu",
            )
            torch.nn.init.kaiming_normal_(
                self._key_matrices[h].weight,
                nonlinearity="relu",
            )
            torch.nn.init.kaiming_normal_(
                self._value_matrices[h].weight,
                nonlinearity="relu",
            )

        torch.nn.init.kaiming_normal_(
            self._output.weight,
            nonlinearity="relu",
        )

        return

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        scores: torch.Tensor = q @ k.transpose(-2, -1) / math.sqrt(self._w_q_k)

        if self._masked:
            mask_dim: int = q.shape[0]
            mask: torch.tensor = torch.tril(torch.ones(mask_dim, mask_dim)).to(
                self._device
            )
            scores = scores.masked_fill(mask == 0, float("-inf"))

        return torch.softmax(scores, dim=-1) @ v
