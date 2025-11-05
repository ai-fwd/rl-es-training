from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Tuple

import cma
import numpy as np

from rlo.evaluators.evaluate import evaluate_candidate
from rlo.policies.param_base import Policy
from rlo.utils.logging import GenerationStats
from rlo.utils.serialization import PolicyBundle


def train_cma_es(
    make_policy: Callable[[], Policy],
    make_features: Callable[[Dict[str, Any], Dict[str, Any], int], np.ndarray],
    generations: int = 60,
    pop_multiplier: float = 1.0,
    init_sigma: float = 0.5,
    base_seed: int = 42,
    horizon: int = 1800,
) -> Tuple[PolicyBundle, list[GenerationStats]]:
    """Trains a policy using CMA-ES in parameter space.

    Args:
        make_policy: A callable that returns a new instance of the policy to be trained.
        make_features: A callable that extracts features from observations.
        generations: Number of CMA-ES generations to run.
        pop_multiplier: Multiplier for the CMA-ES population size.
        init_sigma: Initial standard deviation for CMA-ES.
        base_seed: Base random seed for environment evaluation.
        horizon: Maximum episode length for evaluation.

    Returns:
        A dictionary containing training results and the best found parameters.
    """

    # Calculate the population size. From https://arxiv.org/pdf/1604.00772 pg. 31
    # ✅ It adapts automatically to policy size.
    # ✅ It keeps covariance estimation numerically stable.
    # ✅ It follows empirically validated CMA-ES design principles.
    # ✅ It avoids over- or under-sampling when you change architectures.

    # Initialize policy to get number of parameters/dimensions
    policy = make_policy()
    num_params = policy.num_params()

    # Standard population size: λ = 4 + ⌊3 ln d⌋
    lam = max(4 + int(3 * math.log(num_params)), 6)
    lam = int(max(4, round(pop_multiplier * lam)))

    # Create a starting mean
    rng = np.random.default_rng(base_seed)
    m0 = rng.random(num_params).astype(np.float32) * 0.1  # small random init

    # Initialize CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(
        m0,
        init_sigma,
        {
            "popsize": lam,
            "seed": base_seed,
            #'verb_log': 0,
            #'verb_disp': 1,
        },
    )

    history: list[GenerationStats] = []
    global_best, best_params = -np.inf, None
    for gen in range(generations):
        ask_time = time.perf_counter()  # track time taken for evaluations
        X = es.ask()

        losses = []
        returns = []

        for i, x in enumerate(X):
            # region Seed Generation Rationale
            # ---------------------------------------------------------------------
            # Seed generation rationale:
            # Each (generation, candidate) pair gets a unique deterministic seed
            # so rollouts are reproducible but statistically independent.
            #
            # Why not `base_seed + gen * λ + i`?
            #   - Sequential seeds (e.g. 1000, 1001, 1002...) can yield correlated
            #     RNG states in linear–congruential or xorshift generators, so
            #     nearby worlds may look too similar.
            #
            # Why multiply gen by a prime (997)?
            #   - 997 is co-prime with typical λ values and large enough to
            #     "spread out" seeds across the 32-bit space, decorrelating them.
            #   - Acts as a cheap integer hash: seed = base + 997*gen + i.
            #
            # Effect:
            #   ✅ Deterministic (same run ⇒ same seeds)
            #   ✅ Unique per rollout (no repeats)
            #   ✅ Decorrelated (different generations ≈ independent worlds)
            # ---------------------------------------------------------------------
            # endregion
            seed = base_seed + gen * 997 + i

            # change seed per evaluation for diversity
            ret = evaluate_candidate(
                params=x,
                make_policy=make_policy,
                make_features=make_features,
                horizon=horizon,
                seed=seed,
            )
            returns.append(ret)
            losses.append(-ret)  # CMA-ES minimizes

        es.tell(X, losses)
        elapsed = time.perf_counter() - ask_time  # time taken for evaluations
        es.disp()

        # vectorize all the candidate returns
        ret_array = np.array(returns, dtype=np.float32)
        gen_best = float(ret_array.max())

        # find the global best policy parameters
        if gen_best > global_best:
            global_best = gen_best
            best_params = X[
                int(ret_array.argmax())
            ].copy()  # store a copy so it's not overwritten

        stats = GenerationStats(
            generation=gen,
            wall_time_s=elapsed,
            best_reward=gen_best,
            mean_reward=0,
            median_reward=0,
            reward_std=0,
            sigma=es.sigma,
        )
        history.append(stats)

        print(
            f"[Gen {gen:03d}] pop_mean={ret_array.mean():.2f} "
            f"pop_best={gen_best:.2f} global_best={global_best:.2f}"
        )

        if es.stop():
            print("CMA-ES stopping criteria met. Ending training.")
            break

    # After training, create the best policy for return
    best_policy = make_policy()
    if best_params is not None:
        best_policy.set_params(best_params)

    # create a policy bundle to return
    return PolicyBundle(
        policy=best_policy,
        metadata={
            "best_return": global_best,
            "dim": num_params,
            "popsize": lam,
        },
    ), history
