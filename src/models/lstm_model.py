"""PyTorch LSTM architecture for Phase 3 energy consumption forecasting.

Defines a compact recurrent network that maps a sequence of multivariate
readings to a single continuous ``Electricity_Consumed`` forecast.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EnergyLSTM(nn.Module):
    """LSTM regressor for next-step electricity consumption.

    Expects input tensors shaped ``(batch, seq_len, n_features)`` and
    returns one scalar prediction per sequence using the final LSTM hidden
    state.

    Args:
        input_size: Number of features per timestep (default ``7`` for the
            clean-artifact feature set used in LSTM prep).
        hidden_size: LSTM hidden state width. Defaults to ``64``.
        dropout: Dropout between stacked LSTM layers; active when
            ``num_layers > 1``. Defaults to ``0.2``.
        num_layers: Number of recurrent layers. Defaults to ``1`` to limit
            capacity on the small smart-meter series.
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        dropout: float = 0.2,
        num_layers: int = 1,
    ) -> None:
        """Initialize LSTM backbone and linear prediction head.

        Args:
            input_size: Number of features per timestep.
            hidden_size: LSTM hidden state width.
            dropout: Dropout between stacked LSTM layers.
            num_layers: Number of recurrent layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Predictions of shape ``(batch,)`` — one consumption value per
            sequence.
        """
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


if __name__ == "__main__":
    model = EnergyLSTM(input_size=7, hidden_size=64)
    sample = torch.randn(4, 24, 7)
    output = model(sample)
    assert output.shape == (4,), f"Expected (4,), got {tuple(output.shape)}"
    print("PASS — EnergyLSTM forward shape:", tuple(output.shape))
