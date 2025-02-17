import math
import torch


class FeedForward(torch.nn.Module):
    """
    A position-wise feed-forward module

    This module applies two linear transformations with ReLU activation

    Args:
        model_dim (int): the input and output dimension
        hidden_dim (int): the dimensionality of the hidden layer
        device (torch.device): the torch device
    """

    _w_1: torch.Tensor
    _w_2: torch.Tensor
    _b_1: torch.Tensor
    _b_2: torch.Tensor

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__()

        he_1: float = math.sqrt(2 / model_dim)
        he_2: float = math.sqrt(2 / hidden_dim)

        self._w_1 = torch.nn.Parameter(
            torch.rand(model_dim, hidden_dim, device=device) * he_1,
            requires_grad=True,
        )
        self._b_1 = torch.nn.Parameter(
            torch.zeros(hidden_dim, device=device),
            requires_grad=True,
        )
        self._w_2 = torch.nn.Parameter(
            torch.rand(hidden_dim, model_dim, device=device) * he_2,
            requires_grad=True,
        )
        self._b_2 = torch.nn.Parameter(
            torch.zeros(model_dim, device=device),
            requires_grad=True,
        )

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x @ self._w_1 + self._b_1) @ self._w_2 + self._b_2
