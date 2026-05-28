"""
LSTM-Autoencoder (LSTM-AE) Predictive Module
=============================================
Dual-function model for:
  1. Workload forecasting (k-step ahead prediction)
  2. Anomaly detection via reconstruction error thresholding

Architecture (Section 4.2 of the paper):
  - Encoder: 2-layer stacked LSTM (128 → 64 units) → latent z ∈ R^64
  - Decoder: mirrors encoder
  - Multi-task loss: λ1·MSE_forecast + λ2·MSE_recon + λ3·BCE_anomaly

Reference: Equation (8) in the paper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class LSTMEncoder(nn.Module):
    """Stacked LSTM encoder mapping input sequence to latent z."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: Tuple[int, int] = (128, 64),
        latent_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dims[0], batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.fc_latent = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) — multivariate time series
        Returns:
            z: (batch, latent_dim) — latent representation
            h_seq: (batch, seq_len, hidden_dim) — full hidden sequence for decoder
        """
        h_seq, _ = self.lstm1(x)
        h_seq2, (h_n, _) = self.lstm2(h_seq)
        z = self.fc_latent(h_n[-1])
        return z, h_seq2


class LSTMDecoder(nn.Module):
    """LSTM decoder reconstructing input sequence from latent z."""

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dims: Tuple[int, int] = (64, 128),
        output_dim: int = 5,
        seq_len: int = 30,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.fc_expand = nn.Linear(latent_dim, hidden_dims[0])
        self.lstm1 = nn.LSTM(hidden_dims[0], hidden_dims[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dims[0], hidden_dims[1], batch_first=True)
        self.fc_out = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim)
        Returns:
            x_hat: (batch, seq_len, output_dim) — reconstructed sequence
        """
        # Expand z into a repeated seed sequence
        h0 = self.fc_expand(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        h1, _ = self.lstm1(h0)
        h2, _ = self.lstm2(h1)
        x_hat = self.fc_out(h2)
        return x_hat


class LSTMAutoencoder(nn.Module):
    """
    Joint LSTM-AE for workload forecasting and anomaly detection.

    Multi-task loss (Equation 8):
        L = λ1·MSE(y, ŷ) + λ2·MSE(x, x̂) + λ3·BCE(a, â)
    where:
        λ1=0.5 (forecast), λ2=0.3 (reconstruction), λ3=0.2 (anomaly)
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dims: Tuple[int, int] = (128, 64),
        latent_dim: int = 64,
        forecast_horizon: int = 5,
        seq_len: int = 30,
        lambda1: float = 0.5,
        lambda2: float = 0.3,
        lambda3: float = 0.2,
        anomaly_percentile: float = 99.0,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.seq_len = seq_len
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.anomaly_percentile = anomaly_percentile

        self.encoder = LSTMEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, (hidden_dims[1], hidden_dims[0]), input_dim, seq_len)

        # Forecasting head: latent → k-step forecast
        self.forecast_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, forecast_horizon * input_dim),
        )

        # Anomaly classification head: recon_error scalar → binary
        self.anomaly_head = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Anomaly threshold (learned from validation errors, 99th percentile)
        self.register_buffer("anomaly_threshold", torch.tensor(float("inf")))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            y_hat: (batch, forecast_horizon, input_dim) — workload forecast
            x_hat: (batch, seq_len, input_dim) — reconstruction
            anomaly_prob: (batch, 1) — anomaly probability
        """
        z, _ = self.encoder(x)
        x_hat = self.decoder(z)

        # Forecast
        y_hat = self.forecast_head(z).view(-1, self.forecast_horizon, x.shape[-1])

        # Reconstruction error for anomaly scoring
        recon_error = torch.mean((x - x_hat) ** 2, dim=[1, 2], keepdim=True)  # (B, 1, 1)
        recon_error_flat = recon_error.squeeze(-1)  # (B, 1)
        anomaly_prob = self.anomaly_head(recon_error_flat)

        return y_hat, x_hat, anomaly_prob

    def compute_loss(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        a_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-task loss (Equation 8).
        Args:
            x: (batch, seq_len, input_dim) — input sequence
            y_true: (batch, forecast_horizon, input_dim) — ground-truth future
            a_true: (batch, 1) — binary anomaly label
        """
        y_hat, x_hat, anomaly_prob = self.forward(x)
        loss_forecast = nn.functional.mse_loss(y_hat, y_true)
        loss_recon = nn.functional.mse_loss(x_hat, x)
        loss_anomaly = nn.functional.binary_cross_entropy(anomaly_prob, a_true)
        return self.lambda1 * loss_forecast + self.lambda2 * loss_recon + self.lambda3 * loss_anomaly

    @torch.no_grad()
    def fit_anomaly_threshold(self, val_loader: torch.utils.data.DataLoader) -> float:
        """
        Compute anomaly threshold as the 99th percentile of reconstruction
        errors on the validation set. Called once after training.
        """
        self.eval()
        errors = []
        for batch in val_loader:
            x = batch["x"]
            _, x_hat, _ = self.forward(x)
            recon_err = torch.mean((x - x_hat) ** 2, dim=[1, 2]).cpu().numpy()
            errors.extend(recon_err.tolist())
        threshold = float(np.percentile(errors, self.anomaly_percentile))
        self.anomaly_threshold = torch.tensor(threshold)
        return threshold

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """
        Inference: returns forecast, reconstruction, and anomaly flag.
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            dict with keys: forecast, reconstruction, anomaly_prob, is_anomaly
        """
        self.eval()
        y_hat, x_hat, anomaly_prob = self.forward(x)
        recon_err = torch.mean((x - x_hat) ** 2, dim=[1, 2])
        is_anomaly = recon_err > self.anomaly_threshold
        return {
            "forecast": y_hat,
            "reconstruction": x_hat,
            "anomaly_prob": anomaly_prob,
            "reconstruction_error": recon_err,
            "is_anomaly": is_anomaly,
        }
