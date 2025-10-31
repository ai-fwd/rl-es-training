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

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Action conventions
# ---------------------------------------------------------------------------

# The EndlessPlatformer environment exposes four discrete actions but our
# teaching-focused controller intentionally ignores the jump to keep the search
# space compact.  The indices in ACTION_MAPPING correspond to local policy
# outputs (0, 1, 2) and map onto the environment's discrete action ids.
ACTION_MAPPING: Tuple[int, int, int] = (0, 1, 3)  # NOOP, FORWARD, EAT
ACTION_LABELS: Dict[int, str] = {
    0: "noop",
    1: "forward",
    3: "eat",
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class FeatureExtractorConfig:
    """Configuration bag for :class:`FeatureExtractor`.

    Attributes:
        front_columns: Number of rightmost image columns to summarise.  These
            columns correspond to the agent's forward view.
        floor_rows: Number of rows at the bottom of the image used to estimate
            whether the upcoming tile is solid or a gap.
        energy_delta_decay: Exponential smoothing factor applied to the energy
            derivative feature.  A value in (0, 1] – 1.0 means "no smoothing".
        normalise_elapsed_time: When ``True`` squash the elapsed time into the
            [-1, 1] interval using ``tanh`` to keep magnitudes well-behaved.
    """

    front_columns: int = 12
    floor_rows: int = 6
    energy_delta_decay: float = 0.8
    normalise_elapsed_time: bool = True

    def to_dict(self) -> Dict[str, float]:
        """Serialise to a plain dictionary for JSON storage."""
        return {
            "front_columns": int(self.front_columns),
            "floor_rows": int(self.floor_rows),
            "energy_delta_decay": float(self.energy_delta_decay),
            "normalise_elapsed_time": bool(self.normalise_elapsed_time),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> "FeatureExtractorConfig":
        """Instantiate from a mapping (e.g. data loaded from JSON)."""
        return cls(
            front_columns=int(mapping.get("front_columns", cls.front_columns)),
            floor_rows=int(mapping.get("floor_rows", cls.floor_rows)),
            energy_delta_decay=float(
                mapping.get("energy_delta_decay", cls.energy_delta_decay)
            ),
            normalise_elapsed_time=bool(
                mapping.get("normalise_elapsed_time", cls.normalise_elapsed_time)
            ),
        )


@dataclass
class FeatureExtractor:
    """Transform environment observations into low-dimensional features.

    The current design favours interpretability.  A handful of hand-crafted
    signals capture the agent's energetic state and a coarse understanding of
    the scene directly from the 160x90 grayscale observation.
    """

    config: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)

    def __post_init__(self) -> None:
        # Enumerate feature names in the same order as the numerical vector.
        self._feature_names: Tuple[str, ...] = (
            "bias/constant",
            "energy_norm",
            "energy_deficit",
            "energy_overflow",
            "smoothed_energy_delta",
            "can_eat_flag",
            "front_mean_intensity",
            "front_max_intensity",
            "front_contrast",
            "floor_shadow",
            "elapsed_time_signal",
        )
        # Internal state for temporal features.
        self._prev_energy_norm: float | None = None
        self._smoothed_delta: float = 0.0

    # Public API --------------------------------------------------------- #
    @property
    def feature_names(self) -> Tuple[str, ...]:
        return self._feature_names

    @property
    def size(self) -> int:
        return len(self._feature_names)

    def reset(self) -> None:
        """Reset temporal state before starting a fresh rollout."""
        self._prev_energy_norm = None
        self._smoothed_delta = 0.0

    def extract(self, observation: Mapping[str, np.ndarray], info: Mapping[str, float]) -> np.ndarray:
        """Return the feature vector as ``np.float32``.

        Args:
            observation: Environment observation dictionary containing:
                * ``"image"``: Grayscale image, ``H x W x 1``.
                * ``"metrics"``: Float vector ``[energy, energy_max, elapsed_s]``.
            info: Optional extra signals produced by the environment.  We use
                this to access ``can_eat`` (whether the agent is intersecting an
                orb) and velocity estimates.
        """
        metrics = observation["metrics"].astype(np.float32)
        energy = float(metrics[0])
        energy_max = float(metrics[1])
        elapsed_s = float(metrics[2])

        energy_max = max(energy_max, 1e-6)  # avoid division by zero
        energy_norm = energy / energy_max
        energy_deficit = max(0.0, 1.0 - energy_norm)
        energy_overflow = max(0.0, energy_norm - 1.0)

        prev_energy_norm = self._prev_energy_norm if self._prev_energy_norm is not None else energy_norm
        raw_delta = energy_norm - prev_energy_norm
        decay = float(np.clip(self.config.energy_delta_decay, 0.0, 1.0))
        self._smoothed_delta = decay * self._smoothed_delta + (1.0 - decay) * raw_delta
        self._prev_energy_norm = energy_norm

        can_eat_flag = 1.0 if bool(info.get("can_eat", False)) else 0.0

        image = observation["image"]
        if image.ndim == 3:
            image = image[:, :, 0]
        image = image.astype(np.float32) / 255.0  # normalise intensity

        # Forward view: a slice of rightmost columns.
        front_cols = int(np.clip(self.config.front_columns, 1, image.shape[1]))
        front_view = image[:, -front_cols:]
        front_mean = float(front_view.mean())
        front_max = float(front_view.max())
        front_contrast = float(front_max - front_mean)

        # Estimate whether the upcoming tile is solid by averaging the bottom rows.
        floor_rows = int(np.clip(self.config.floor_rows, 1, image.shape[0]))
        floor_strip = image[-floor_rows:, :]
        # When the strip is darker the agent is likely above a gap.
        floor_shadow = float(1.0 - floor_strip.mean())

        if self.config.normalise_elapsed_time:
            elapsed_signal = float(np.tanh(elapsed_s / 60.0))
        else:
            elapsed_signal = elapsed_s

        features = np.array(
            [
                1.0,  # bias term
                energy_norm,
                energy_deficit,
                energy_overflow,
                self._smoothed_delta,
                can_eat_flag,
                front_mean,
                front_max,
                front_contrast,
                floor_shadow,
                elapsed_signal,
            ],
            dtype=np.float32,
        )
        return features


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
        weights = rng.normal(loc=0.0, scale=weight_scale, size=(n_actions, n_features)).astype(np.float32)
        bias = np.zeros((n_actions,), dtype=np.float32)
        return cls(weights=weights, bias=bias, action_ids=tuple(action_ids), feature_names=tuple(feature_names))

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
        return cls(weights=weights, bias=bias, action_ids=tuple(action_ids), feature_names=tuple(feature_names))

    def to_flat(self) -> np.ndarray:
        """Flatten parameters into a single 1-D vector."""
        return np.concatenate([self.weights.flatten(), self.bias], axis=0)

    # Action selection ---------------------------------------------------- #
    def logits(self, features: np.ndarray) -> np.ndarray:
        """Return raw logits for each action."""
        features = np.asarray(features, dtype=np.float32).ravel()
        return self.weights @ features + self.bias

    def action_probabilities(self, features: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Return softmax probabilities with optional temperature scaling."""
        logits = self.logits(features)
        logits = logits / max(temperature, 1e-6)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits.astype(np.float64))
        probs = exp_logits / exp_logits.sum()
        return probs.astype(np.float32)

    def select_action(self, features: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict[str, np.ndarray]]:
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

@dataclass
class PolicyBundle:
    """A convenience wrapper bundling a policy with metadata and config."""

    policy: LinearPolicy
    feature_config: FeatureExtractorConfig
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, object]:
        """Convert bundle into JSON/NumPy friendly payload."""
        return {
            "weights": self.policy.weights,
            "bias": self.policy.bias,
            "action_ids": np.array(self.policy.action_ids, dtype=np.int32),
            "feature_names": np.array(self.policy.feature_names, dtype=object),
            "feature_config": json.dumps(self.feature_config.to_dict()),
            "metadata": json.dumps(self.metadata),
        }

    def save(self, path: str | Path) -> Path:
        """Persist the bundle to ``.npz`` format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload()
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PolicyBundle":
        """Load a previously saved policy bundle."""
        data = np.load(Path(path), allow_pickle=True)
        weights = data["weights"]
        bias = data["bias"]
        action_ids = tuple(int(x) for x in data["action_ids"].tolist())
        feature_names = tuple(str(x) for x in data["feature_names"].tolist())
        feature_cfg_raw = json.loads(data["feature_config"].item())
        feature_cfg = FeatureExtractorConfig.from_mapping(feature_cfg_raw)
        metadata = json.loads(data["metadata"].item())
        policy = LinearPolicy(weights=weights, bias=bias, action_ids=action_ids, feature_names=feature_names)
        return cls(policy=policy, feature_config=feature_cfg, metadata=metadata)


def load_policy_bundle(path: str | Path) -> PolicyBundle:
    """Shim for convenient imports."""
    return PolicyBundle.load(path)
