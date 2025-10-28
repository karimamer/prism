"""
Confidence Calibration for ReLiK.

Implements temperature scaling and Platt scaling to calibrate
confidence scores for better probability interpretation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    """
    Temperature scaling for confidence calibration.

    Learns a single temperature parameter to scale logits before softmax.
    Simple but effective method from Guo et al. (2017).
    """

    def __init__(self, initial_temperature: float = 1.0):
        """
        Initialize temperature scaler.

        Args:
            initial_temperature: Initial temperature value
        """
        super().__init__()

        # Temperature parameter (learned)
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale logits by temperature.

        Args:
            logits: Raw model logits [batch, num_classes]

        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ):
        """
        Fit temperature parameter using validation data.

        Args:
            logits: Validation logits [num_samples, num_classes]
            labels: Ground truth labels [num_samples]
            max_iter: Maximum optimization iterations
            lr: Learning rate
        """
        # Create optimizer
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            """Compute NLL loss."""
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            # cross_entropy expects Long labels for class indices
            labels_long = labels.long() if labels.dtype != torch.long else labels
            loss = F.cross_entropy(scaled_logits, labels_long)
            loss.backward()
            return loss

        # Optimize
        optimizer.step(eval_loss)

        # Clamp temperature to reasonable range
        with torch.no_grad():
            self.temperature.clamp_(0.1, 10.0)


class PlattScaler(nn.Module):
    """
    Platt scaling for binary confidence calibration.

    Fits a logistic regression: P(y=1|x) = 1 / (1 + exp(A*x + B))
    where x is the model's output score.
    """

    def __init__(self):
        """Initialize Platt scaler."""
        super().__init__()

        # Logistic regression parameters
        self.A = nn.Parameter(torch.ones(1))
        self.B = nn.Parameter(torch.zeros(1))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply Platt scaling to scores.

        Args:
            scores: Raw model scores [batch]

        Returns:
            Calibrated probabilities [batch]
        """
        return torch.sigmoid(self.A * scores + self.B)

    def fit(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
    ):
        """
        Fit Platt scaling parameters.

        Args:
            scores: Validation scores [num_samples]
            labels: Binary ground truth labels [num_samples]
            max_iter: Maximum optimization iterations
            lr: Learning rate
        """
        optimizer = torch.optim.Adam([self.A, self.B], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()

            # Compute calibrated probabilities
            probs = self.forward(scores)

            # Binary cross-entropy loss
            loss = F.binary_cross_entropy(probs, labels.float())

            loss.backward()
            optimizer.step()


class ConfidenceCalibrator:
    """
    Unified confidence calibrator for ReLiK components.

    Calibrates:
    1. Span detection scores
    2. Entity linking scores
    3. Relation extraction scores
    """

    def __init__(self, method: str = "temperature"):
        """
        Initialize calibrator.

        Args:
            method: Calibration method ('temperature' or 'platt')
        """
        self.method = method

        # Calibrators for different components
        self.span_calibrator: Optional[nn.Module] = None
        self.entity_calibrator: Optional[nn.Module] = None
        self.relation_calibrator: Optional[nn.Module] = None

    def fit_span_calibrator(
        self,
        span_logits: torch.Tensor,
        span_labels: torch.Tensor,
    ):
        """
        Fit span detection calibrator.

        Args:
            span_logits: Span detection logits from validation set
            span_labels: Ground truth span labels
        """
        if self.method == "temperature":
            self.span_calibrator = TemperatureScaler()
            self.span_calibrator.fit(span_logits, span_labels)
        elif self.method == "platt":
            # For Platt, we need binary scores
            span_scores = torch.sigmoid(span_logits)
            if span_scores.ndim > 1:
                span_scores = span_scores[:, 1]  # Positive class
            self.span_calibrator = PlattScaler()
            self.span_calibrator.fit(span_scores, span_labels)

    def fit_entity_calibrator(
        self,
        entity_scores: torch.Tensor,
        entity_labels: torch.Tensor,
    ):
        """
        Fit entity linking calibrator.

        Args:
            entity_scores: Entity linking scores from validation set
            entity_labels: Ground truth entity labels (0/1)
        """
        if self.method == "platt":
            self.entity_calibrator = PlattScaler()
            self.entity_calibrator.fit(entity_scores, entity_labels)
        else:
            # Temperature scaling for multi-class
            self.entity_calibrator = TemperatureScaler()
            # Convert 1D binary scores to 2D logits for temperature scaling
            if entity_scores.dim() == 1:
                # Binary case: convert to [batch, 2] logits
                entity_logits = torch.stack([-entity_scores, entity_scores], dim=1)
            else:
                entity_logits = entity_scores
            self.entity_calibrator.fit(entity_logits, entity_labels)

    def fit_relation_calibrator(
        self,
        relation_scores: torch.Tensor,
        relation_labels: torch.Tensor,
    ):
        """
        Fit relation extraction calibrator.

        Args:
            relation_scores: Relation scores from validation set
            relation_labels: Ground truth relation labels (0/1)
        """
        if self.method == "platt":
            self.relation_calibrator = PlattScaler()
            self.relation_calibrator.fit(relation_scores, relation_labels)
        else:
            self.relation_calibrator = TemperatureScaler()
            self.relation_calibrator.fit(relation_scores, relation_labels)

    def calibrate_span_scores(
        self,
        span_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calibrate span detection scores.

        Args:
            span_logits: Raw span logits

        Returns:
            Calibrated probabilities
        """
        if self.span_calibrator is None:
            # No calibration, just apply sigmoid
            return torch.sigmoid(span_logits)

        if self.method == "temperature":
            calibrated_logits = self.span_calibrator(span_logits)
            return torch.softmax(calibrated_logits, dim=-1)
        else:  # platt
            span_scores = torch.sigmoid(span_logits)
            if span_scores.ndim > 1:
                span_scores = span_scores[:, 1]
            return self.span_calibrator(span_scores)

    def calibrate_entity_scores(
        self,
        entity_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calibrate entity linking scores.

        Args:
            entity_scores: Raw entity scores

        Returns:
            Calibrated probabilities
        """
        if self.entity_calibrator is None:
            return entity_scores

        if self.method == "platt":
            return self.entity_calibrator(entity_scores)
        else:
            calibrated_logits = self.entity_calibrator(entity_scores)
            return torch.softmax(calibrated_logits, dim=-1)

    def calibrate_relation_scores(
        self,
        relation_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calibrate relation extraction scores.

        Args:
            relation_scores: Raw relation scores

        Returns:
            Calibrated probabilities
        """
        if self.relation_calibrator is None:
            return relation_scores

        if self.method == "platt":
            return self.relation_calibrator(relation_scores)
        else:
            calibrated_logits = self.relation_calibrator(relation_scores)
            return torch.softmax(calibrated_logits, dim=-1)

    def save(self, path: str):
        """Save calibrators to disk."""
        import os

        os.makedirs(path, exist_ok=True)

        if self.span_calibrator is not None:
            torch.save(self.span_calibrator.state_dict(), f"{path}/span_calibrator.pt")

        if self.entity_calibrator is not None:
            torch.save(self.entity_calibrator.state_dict(), f"{path}/entity_calibrator.pt")

        if self.relation_calibrator is not None:
            torch.save(self.relation_calibrator.state_dict(), f"{path}/relation_calibrator.pt")

        # Save metadata
        import json

        metadata = {"method": self.method}
        with open(f"{path}/calibration_metadata.json", "w") as f:
            json.dump(metadata, f)

    def load(self, path: str):
        """Load calibrators from disk."""
        import json
        import os

        # Load metadata
        with open(f"{path}/calibration_metadata.json") as f:
            metadata = json.load(f)

        self.method = metadata["method"]

        # Load calibrators
        if os.path.exists(f"{path}/span_calibrator.pt"):
            if self.method == "temperature":
                self.span_calibrator = TemperatureScaler()
            else:
                self.span_calibrator = PlattScaler()
            self.span_calibrator.load_state_dict(torch.load(f"{path}/span_calibrator.pt"))

        if os.path.exists(f"{path}/entity_calibrator.pt"):
            if self.method == "temperature":
                self.entity_calibrator = TemperatureScaler()
            else:
                self.entity_calibrator = PlattScaler()
            self.entity_calibrator.load_state_dict(torch.load(f"{path}/entity_calibrator.pt"))

        if os.path.exists(f"{path}/relation_calibrator.pt"):
            if self.method == "temperature":
                self.relation_calibrator = TemperatureScaler()
            else:
                self.relation_calibrator = PlattScaler()
            self.relation_calibrator.load_state_dict(torch.load(f"{path}/relation_calibrator.pt"))


__all__ = ["TemperatureScaler", "PlattScaler", "ConfidenceCalibrator"]
