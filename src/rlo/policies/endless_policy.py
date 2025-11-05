"""Shared policy definitions and utilities for the Endless Platformer agent.

This module intentionally focuses on *clarity* over raw performance.  The goal
is to expose how a simple linear controller can be optimised with evolution
strategies (CMA-ES) and later reused for interactive visualisations.

Key concepts implemented here:

* :class:`FeatureExtractor` transforms rich environment observations into a
  compact, interpretable feature vector.  The implementation keeps intermediate
  values around so we can display them in teaching tools (e.g. a scratchpad UI).
* :class:`LinearPolicy` represents a differentiable linear (affine) controller
  whose parameters are tuned by CMA-ES.  Helper methods expose action logits,
  probabilities, and per-feature contribution scores.
* :class:`PolicyBundle` wires both pieces together and knows how to (de-)serialise
  policies so they can be trained once and later visualised or deployed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from rlo.utils.serialization import PolicyBundle

# ---------------------------------------------------------------------------
# Linear policy representation
# ---------------------------------------------------------------------------


@dataclass
class LinearPolicy:
    """Simple affine controller (logits = W · x + b).

    The CMA-ES algorithm optimises the flattened parameter vector.  We expose
    utilities to convert between the flat representation and structured weights.
    """

    weights: np.ndarray  # shape: (n_actions, n_features)
    bias: np.ndarray  # shape: (n_actions,)
    action_ids: Tuple[int, ...]
    feature_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float32)
        self.bias = np.asarray(self.bias, dtype=np.float32)
        if self.weights.ndim != 2:
            raise ValueError("weights must be a 2D array [actions, features].")
        if self.bias.shape != (self.weights.shape[0],):
            raise ValueError("bias must have shape (n_actions,).")
        if len(self.action_ids) != self.weights.shape[0]:
            raise ValueError("action_ids length must match number of actions.")
        if len(self.feature_names) != self.weights.shape[1]:
            raise ValueError("feature_names length must match number of features.")

    # Construction helpers ------------------------------------------------ #
    @classmethod
    def initialise(
        cls,
        n_actions: int,
        n_features: int,
        action_ids: Sequence[int],
        feature_names: Sequence[str],
        weight_scale: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> "LinearPolicy":
        """Factory method that initialises small random parameters."""
        rng = rng or np.random.default_rng()
        weights = rng.normal(
            loc=0.0, scale=weight_scale, size=(n_actions, n_features)
        ).astype(np.float32)
        bias = np.zeros((n_actions,), dtype=np.float32)
        return cls(
            weights=weights,
            bias=bias,
            action_ids=tuple(action_ids),
            feature_names=tuple(feature_names),
        )

    @classmethod
    def from_flat(
        cls,
        flat: np.ndarray,
        n_actions: int,
        n_features: int,
        action_ids: Sequence[int],
        feature_names: Sequence[str],
    ) -> "LinearPolicy":
        """Reconstruct parameters from a 1-D CMA-ES chromosome."""
        expected = n_actions * n_features + n_actions
        flat = np.asarray(flat, dtype=np.float32).ravel()
        if flat.size != expected:
            raise ValueError(f"Flat vector has size {flat.size}, expected {expected}.")
        weights = flat[: n_actions * n_features].reshape(n_actions, n_features)
        bias = flat[n_actions * n_features :]
        return cls(
            weights=weights,
            bias=bias,
            action_ids=tuple(action_ids),
            feature_names=tuple(feature_names),
        )

    def to_flat(self) -> np.ndarray:
        """Flatten parameters into a single 1-D vector."""
        return np.concatenate([self.weights.flatten(), self.bias], axis=0)

    # Action selection ---------------------------------------------------- #
    def logits(self, features: np.ndarray) -> np.ndarray:
        """Return raw logits for each action."""
        features = np.asarray(features, dtype=np.float32).ravel()
        return self.weights @ features + self.bias

    def action_probabilities(
        self, features: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        """Return softmax probabilities with optional temperature scaling."""
        logits = self.logits(features)
        logits = logits / max(temperature, 1e-6)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits.astype(np.float64))
        probs = exp_logits / exp_logits.sum()
        return probs.astype(np.float32)

    def select_action(
        self, features: np.ndarray, deterministic: bool = True
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """Select an action id and provide auxiliary diagnostics.

        Args:
            features: Feature vector for the current timestep.
            deterministic: If ``True`` choose the argmax action; otherwise
                sample according to the action probabilities.
        Returns:
            action_id: Environment action.
            diagnostics: Dict with logits, probabilities, and contributions.
        """
        probs = self.action_probabilities(features)
        logits = self.logits(features)

        if deterministic:
            local_action = int(np.argmax(probs))
        else:
            local_action = int(np.random.choice(len(probs), p=probs))

        contributions = self.feature_contributions(features)
        info = {
            "local_index": local_action,
            "logits": logits,
            "probabilities": probs,
            "contributions": contributions,
        }
        return self.action_ids[local_action], info

    def feature_contributions(self, features: np.ndarray) -> np.ndarray:
        """Return per-feature contributions (weights * feature value)."""
        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return self.weights * features  # broadcasting


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def load_policy_bundle(path: str | Path) -> PolicyBundle:
    """Shim for convenient imports."""
    return PolicyBundle.load(path)
