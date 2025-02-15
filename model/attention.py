import math
import torch


class Attention(torch.nn.Module):
    """
    A trainable attention module

    Args:
        model_dim: the input and output dimension
        w_q_k: the query/key dimension
        w_v: the value dimension
    """

    _query: torch.nn.Linear
    _key: torch.nn.Linear
    _value: torch.nn.Linear
    _w_q_k: int

    def __init__(self, model_dim: int, w_q_k: int, w_v: int) -> None:
        super().__init__()

        # Bias not used in the paper
        self._query = torch.nn.Linear(model_dim, w_q_k, bias=False)
        self._key = torch.nn.Linear(model_dim, w_q_k, bias=False)
        self._value = torch.nn.Linear(model_dim, w_v, bias=False)

        self._w_q_k = w_q_k

        # We opt for He initialization, as the linear layers are followed by a ReLU non-linearity as per the paper
        torch.nn.init.kaiming_normal_(self._query.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self._key.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self._value.weight, nonlinearity="relu")

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._query(x)
        k = self._key(x)
        v = self._value(x)

        scores: torch.Tensor = q @ k.transpose(-2, -1) / math.sqrt(self._w_q_k)
        attention: torch.Tensor = torch.softmax(scores, dim=-1)

        return attention @ v
