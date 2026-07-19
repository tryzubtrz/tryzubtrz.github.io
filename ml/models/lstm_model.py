"""LSTM price-direction model using PyTorch."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import MODELS_DIR

logger = logging.getLogger("ai.lstm")


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),  # short / flat / long → classes 0,1,2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMModel:
    def __init__(
        self,
        n_features: int = 20,
        seq_len: int = 32,
        device: Optional[str] = None,
    ) -> None:
        self.n_features = n_features
        self.seq_len = seq_len
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = LSTMClassifier(n_features).to(self.device)
        self.trained = False
        self.metrics: Dict[str, float] = {}

    @staticmethod
    def _to_class_index(y: np.ndarray) -> np.ndarray:
        # -1,0,1 → 0,1,2
        return (y + 1).astype(np.int64)

    @staticmethod
    def _from_class_index(idx: int) -> int:
        return int(idx - 1)

    def fit(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        epochs: int = 15,
        batch_size: int = 64,
        lr: float = 1e-3,
        val_split: float = 0.15,
        warm_start: bool = True,
    ) -> Dict[str, float]:
        """
        Continual learning: keep existing weights when warm_start=True
        (only rebuild network if feature count changed).
        """
        if len(X_seq) < 50:
            logger.warning("Not enough LSTM samples (%s)", len(X_seq))
            return {"accuracy": 0.0, "samples": float(len(X_seq))}

        n_features = int(X_seq.shape[-1])
        continued = False
        if warm_start and self.trained and self.n_features == n_features:
            # Continue from previous brain — do NOT reset weights
            continued = True
            lr = min(lr, 3e-4)  # finer steps when improving an existing brain
            logger.info("LSTM warm-start from previous brain lr=%s", lr)
        else:
            self.n_features = n_features
            self.model = LSTMClassifier(n_features).to(self.device)

        y_idx = self._to_class_index(y)
        n = len(X_seq)
        split = int(n * (1 - val_split))
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_idx[:split], y_idx[split:]

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
            logger.debug("LSTM epoch %s loss=%.4f", epoch + 1, total_loss / max(len(loader), 1))

        acc = self._eval_accuracy(X_val, y_val)
        self.trained = True
        self.metrics = {
            "accuracy": acc,
            "samples": float(n),
            "epochs": float(epochs),
            "warm_start": float(1.0 if continued else 0.0),
        }
        logger.info(
            "LSTM trained acc=%.3f samples=%s device=%s warm_start=%s",
            acc,
            n,
            self.device,
            continued,
        )
        return self.metrics

    def _eval_accuracy(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        if len(X_val) == 0:
            return 0.0
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_val, dtype=torch.float32).to(self.device))
            pred = logits.argmax(dim=1).cpu().numpy()
        return float((pred == y_val).mean())

    def predict_proba_direction(self, X_seq: np.ndarray) -> Tuple[str, float]:
        """Return direction and confidence for the last sequence window."""
        if len(X_seq) == 0:
            return "flat", 0.0
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(X_seq[-1:], dtype=torch.float32).to(self.device)
            probs = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        direction = {0: "short", 1: "flat", 2: "long"}[idx]
        return direction, float(probs[idx])

    def save(self, path: Path | None = None) -> Path:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = path or (MODELS_DIR / "lstm_latest.pt")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "n_features": self.n_features,
                "seq_len": self.seq_len,
                "metrics": self.metrics,
            },
            path,
        )
        return path

    def load(self, path: Path | None = None) -> bool:
        path = path or (MODELS_DIR / "lstm_latest.pt")
        if not path.exists():
            return False
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.n_features = blob.get("n_features", self.n_features)
        self.seq_len = blob.get("seq_len", self.seq_len)
        self.model = LSTMClassifier(self.n_features).to(self.device)
        self.model.load_state_dict(blob["state_dict"])
        self.metrics = blob.get("metrics", {})
        self.trained = True
        return True
