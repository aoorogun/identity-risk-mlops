from __future__ import annotations

from typing import Any


def build_multitask_network(input_dim: int) -> Any:
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("Install .[deep-learning]") from exc

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 48), nn.ReLU()
            )
            self.failure = nn.Linear(48, 1)
            self.latency = nn.Linear(48, 1)

        def forward(self, x):
            hidden = self.shared(x)
            return self.failure(hidden), self.latency(hidden)

    return Network()
