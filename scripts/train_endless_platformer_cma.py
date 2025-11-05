#!/usr/bin/env python3
"""Train a simple Endless Platformer agent with CMA-ES.

The script demonstrates an Evolution Strategy workflow end-to-end:

1. Convert high-dimensional observations into a compact, interpretable feature
   vector (:class:`rlo.agents.endless_policy.FeatureExtractor`).
2. Optimise a linear controller with `cma.CMAEvolutionStrategy`.
3. Track learning progress, persist the best policy, and optionally create
   visual artefacts that make the optimisation process tangible.

The defaults favour *clarity* over raw performance.  Expect training to finish
within a few minutes on a typical laptop while producing console logs and files
that you can inspect later:

    runs/endless_cma/
        history.json          -> raw per-generation statistics
        best_policy.npz       -> policy bundle loadable by the viewer
        episode_trace.json    -> detailed trace of the best policy (optional)
        reward_plot.png       -> reward curve if matplotlib is available

Usage example (after installing ``pip install cma matplotlib``):

    $ python -m scripts.train_endless_platformer_cma --generations 40 --save-dir runs/demo
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from rlo.utils.feature_extractors import BasicFeatureExtractor

from rlo.policies.param_linear import ACTION_LABELS, ACTION_MAPPING
from rlo.utils.serialization import PolicyBundle

matplotlib.use("Agg")  # Use non-GUI backend

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is optional
    plt = None  # type: ignore

import cma

from rlo.envs.endless_platformer import EndlessPlatformerEnv
from rlo.policies.endless_policy import (
    FeatureExtractorConfig,
    LinearPolicy,
)
from rlo.utils.logging import GenerationStats

# ---------------------------------------------------------------------------
# Configuration data classes
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """High-level knobs that control the CMA-ES run."""

    generations: int = 40
    population_size: Optional[int] = None  # Let CMA choose a default when None.
    episodes_per_candidate: int = 4
    max_steps_per_episode: int = 1800
    sigma0: float = 0.6  # Initial search radius in parameter space.
    seed: Optional[int] = None
    save_dir: Path = Path("runs/endless_cma")
    record_best_trace: bool = True
    gap_probability: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["save_dir"] = str(self.save_dir)
        return data


# ---------------------------------------------------------------------------
# Rollout utilities
# ---------------------------------------------------------------------------


def _action_name(action_id: int) -> str:
    return ACTION_LABELS.get(action_id, f"id={action_id}")


@dataclass
class EpisodeTrace:
    """Detailed per-step record used for post-hoc visualisations."""

    total_reward: float
    steps: int
    terminated: bool
    records: List[Dict[str, object]]

    def to_jsonable(self) -> Dict[str, object]:
        return {
            "total_reward": float(self.total_reward),
            "steps": int(self.steps),
            "terminated": bool(self.terminated),
            "records": self.records,
        }


class RolloutRunner:
    """Helper that runs policy rollouts with consistent feature extraction."""

    def __init__(
        self,
        env: EndlessPlatformerEnv,
        feature_extractor: BasicFeatureExtractor,
        max_steps: int,
    ) -> None:
        self._env = env
        self._extractor = feature_extractor
        self._max_steps = max_steps

    def run(
        self,
        policy: LinearPolicy,
        *,
        deterministic: bool = True,
        record: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[float, EpisodeTrace | None]:
        """Execute a single episode and optionally capture a trace."""
        observation, info = self._env.reset(seed=seed)
        self._extractor.reset()
        features = self._extractor.extract(observation, info)
        total_reward = 0.0
        trace_records: List[Dict[str, object]] = []

        for step in range(self._max_steps):
            action_id, diag = policy.select_action(
                features, deterministic=deterministic
            )
            next_obs, reward, terminated, truncated, next_info = self._env.step(
                action_id
            )
            total_reward += float(reward)

            if record:
                trace_records.append(
                    {
                        "step": step,
                        "action": int(action_id),
                        "action_label": _action_name(action_id),
                        "reward": float(reward),
                        "cumulative_reward": float(total_reward),
                        "features": features.astype(float).tolist(),
                        "feature_names": list(policy.feature_names),
                        "logits": diag["logits"].astype(float).tolist(),
                        "probabilities": diag["probabilities"].astype(float).tolist(),
                        "contributions": diag["contributions"].astype(float).tolist(),
                        "info": {
                            "x_position": float(next_info.get("x_position", math.nan)),
                            "y_position": float(next_info.get("y_position", math.nan)),
                            "on_ground": bool(next_info.get("on_ground", False)),
                            "can_eat": bool(next_info.get("can_eat", False)),
                            "energy": float(next_info.get("energy", math.nan)),
                        },
                    }
                )

            if terminated or truncated:
                self._extractor.reset()
                trace = (
                    EpisodeTrace(
                        total_reward=total_reward,
                        steps=step + 1,
                        terminated=terminated,
                        records=trace_records,
                    )
                    if record
                    else None
                )
                return total_reward, trace

            observation = next_obs
            info = next_info
            features = self._extractor.extract(observation, info)

        # Episode finished because we reached max_steps.
        self._extractor.reset()
        trace = (
            EpisodeTrace(
                total_reward=total_reward,
                steps=self._max_steps,
                terminated=False,
                records=trace_records,
            )
            if record
            else None
        )
        return total_reward, trace


# ---------------------------------------------------------------------------
# CMA-ES trainer
# ---------------------------------------------------------------------------


class CMAESTrainer:
    """Encapsulates CMA-ES optimisation for the Endless Platformer agent."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.save_dir = config.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.feature_extractor = BasicFeatureExtractor(FeatureExtractorConfig())
        self.n_features = self.feature_extractor.size
        self.action_ids = ACTION_MAPPING
        self.n_actions = len(self.action_ids)

        rng = np.random.default_rng(config.seed)
        init_policy = LinearPolicy.initialise(
            n_actions=self.n_actions,
            n_features=self.n_features,
            action_ids=self.action_ids,
            feature_names=self.feature_extractor.feature_names,
            weight_scale=0.05,
            rng=rng,
        )
        self.initial_vector = init_policy.to_flat()

        es_options = {
            "seed": config.seed,
            "popsize": config.population_size,
            "maxiter": config.generations,
            "verb_disp": 0,  # We manage console output ourselves.
        }
        self.es = cma.CMAEvolutionStrategy(
            self.initial_vector, config.sigma0, es_options
        )
        self.best_reward = -math.inf
        self.best_policy_vector = self.initial_vector.copy()
        self.history: List[GenerationStats] = []

    # -------- Evaluation ------------------------------------------------ #
    def _make_env(self) -> EndlessPlatformerEnv:
        return EndlessPlatformerEnv(
            render_mode=None,
            gap_probability=self.config.gap_probability,
        )

    def evaluate_candidate(self, vector: np.ndarray, seeds: Sequence[int]) -> float:
        """Evaluate a single CMA-ES candidate; return negative mean reward."""
        policy = LinearPolicy.from_flat(
            vector,
            n_actions=self.n_actions,
            n_features=self.n_features,
            action_ids=self.action_ids,
            feature_names=self.feature_extractor.feature_names,
        )
        rewards: List[float] = []
        env = self._make_env()
        runner = RolloutRunner(
            env, self.feature_extractor, self.config.max_steps_per_episode
        )
        try:
            for ep_seed in seeds:
                reward, _ = runner.run(
                    policy, deterministic=True, record=False, seed=ep_seed
                )
                rewards.append(reward)
        finally:
            env.close()
        mean_reward = float(np.mean(rewards)) if rewards else -math.inf
        return -mean_reward  # CMA-ES minimises

    # -------- Training loop -------------------------------------------- #
    def train(self) -> PolicyBundle:
        """Execute the CMA-ES loop and return the best policy bundle."""
        cfg = self.config
        episode_rng = np.random.default_rng(cfg.seed)

        for generation in range(cfg.generations):
            ask_time = time.perf_counter()
            candidates = self.es.ask()

            fitness_values = []
            rewards_this_gen = []
            for idx, vector in enumerate(candidates):
                episode_seeds = episode_rng.integers(
                    low=0,
                    high=2**31 - 1,
                    size=cfg.episodes_per_candidate,
                    dtype=np.int64,
                ).tolist()
                fitness = self.evaluate_candidate(vector, episode_seeds)
                fitness_values.append(fitness)
                rewards_this_gen.append(-fitness)

            self.es.tell(candidates, fitness_values)
            elapsed = time.perf_counter() - ask_time
            sigma = float(self.es.sigma)

            rewards_array = np.array(rewards_this_gen, dtype=np.float32)
            best_idx = int(np.argmax(rewards_array))
            best_reward_gen = float(rewards_array[best_idx])
            mean_reward = float(rewards_array.mean())
            median_reward = float(np.median(rewards_array))
            std_reward = float(rewards_array.std(ddof=0))

            if best_reward_gen > self.best_reward:
                self.best_reward = best_reward_gen
                self.best_policy_vector = np.array(candidates[best_idx], copy=True)

            stats = GenerationStats(
                generation=generation,
                wall_time_s=elapsed,
                best_reward=best_reward_gen,
                mean_reward=mean_reward,
                median_reward=median_reward,
                reward_std=std_reward,
                sigma=sigma,
            )
            self.history.append(stats)
            self._log_generation(stats)

            if self.es.stop():
                print("CMA-ES reported stop criteria:", self.es.stop())
                break

        best_policy = LinearPolicy.from_flat(
            self.best_policy_vector,
            n_actions=self.n_actions,
            n_features=self.n_features,
            action_ids=self.action_ids,
            feature_names=self.feature_extractor.feature_names,
        )
        bundle = PolicyBundle(
            policy=best_policy,
            feature_config=self.feature_extractor.config,
            metadata={
                "training_config": self.config.to_dict(),
                "history": [s.to_dict() for s in self.history],
                "best_reward": float(self.best_reward),
            },
        )
        return bundle

    # ------------------------------------------------------------------ #
    def _log_generation(self, stats: GenerationStats) -> None:
        print(
            f"[gen {stats.generation:03d}] "
            f"best={stats.best_reward:7.2f} "
            f"mean={stats.mean_reward:7.2f} "
            f"std={stats.reward_std:6.2f} "
            f"sigma={stats.sigma:5.3f} "
            f"time={stats.wall_time_s:5.2f}s"
        )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def try_plot_history(history: List[GenerationStats], path: Path) -> Optional[Path]:
    if not history or plt is None:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    gens = [s.generation for s in history]
    best = [s.best_reward for s in history]
    mean = [s.mean_reward for s in history]

    ax.plot(gens, mean, label="mean reward", color="#4f81bd")
    ax.plot(gens, best, label="best reward", color="#c05020")
    ax.fill_between(
        gens,
        [m - s.reward_std for m, s in zip(mean, history)],
        [m + s.reward_std for m, s in zip(mean, history)],
        color="#4f81bd",
        alpha=0.2,
        label="±1 std",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Episode reward (distance travelled)")
    ax.set_title("CMA-ES training progress")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise a linear Endless Platformer policy with CMA-ES.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=40,
        help="Number of CMA-ES generations to run.",
    )
    parser.add_argument(
        "--population", type=int, default=None, help="Override CMA-ES population size."
    )
    parser.add_argument(
        "--episodes-per-candidate",
        type=int,
        default=4,
        help="Rollouts per chromosome evaluation.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1800, help="Maximum steps per rollout."
    )
    parser.add_argument(
        "--sigma0",
        type=float,
        default=0.6,
        help="Initial CMA-ES step size (search radius).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for repeatability."
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("runs/endless_cma"),
        help="Directory where artefacts (policy, history, plots) are stored.",
    )
    parser.add_argument(
        "--gap-probability",
        type=float,
        default=0.0,
        help="Probability of generating gaps. Start with 0.0 while learning the basics.",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Skip recording a detailed trace for the best policy (saves time).",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        generations=args.generations,
        population_size=args.population,
        episodes_per_candidate=args.episodes_per_candidate,
        max_steps_per_episode=args.max_steps,
        sigma0=args.sigma0,
        seed=args.seed,
        save_dir=args.save_dir,
        record_best_trace=not args.no_trace,
        gap_probability=args.gap_probability,
    )

    trainer = CMAESTrainer(config)
    bundle = trainer.train()

    # Persist artefacts -------------------------------------------------- #
    history_path = config.save_dir / "history.json"
    write_json(
        history_path,
        {
            "training_config": config.to_dict(),
            "history": [s.to_dict() for s in trainer.history],
        },
    )
    print(f"Wrote history to {history_path}")

    policy_path = config.save_dir / "best_policy.npz"
    bundle.save(policy_path)
    print(f"Saved best policy bundle to {policy_path}")

    plot_path = config.save_dir / "reward_plot.png"
    plotted = try_plot_history(trainer.history, plot_path)
    if plotted:
        print(f"Saved reward plot to {plot_path}")
    else:
        print("Matplotlib not available; skipped reward plot.")

    if config.record_best_trace:
        env = trainer._make_env()
        runner = RolloutRunner(
            env, trainer.feature_extractor, config.max_steps_per_episode
        )
        reward, trace = runner.run(
            bundle.policy, deterministic=True, record=True, seed=config.seed
        )
        env.close()
        trace_path = config.save_dir / "episode_trace.json"
        write_json(trace_path, trace.to_jsonable() if trace else {})
        print(f"Recorded trace (reward={reward:.2f}) to {trace_path}")

    print("\nNext steps:")
    print("  1. Inspect runs in the save directory (history.json, reward_plot.png).")
    print(
        "  2. Launch scripts/play_endless_platformer.py --policy runs/.../best_policy.npz"
    )


if __name__ == "__main__":
    main()
