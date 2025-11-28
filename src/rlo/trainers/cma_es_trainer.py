from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Tuple, cast

import cma
import numpy as np


from rlo.envs.endless_platformer import EndlessPlatformerEnv
from rlo.evaluators.evaluate import evaluate_candidate
from rlo.policies.param_base import Policy
from rlo.utils.logging import GenerationStats
from rlo.utils.serialization import PolicyBundle

from rlo.policies.jepa import JEPAModule
from rlo.policies.param_nonlinear_jepa import ParamNonLinearPolicy_JEPA
import torch
import random
from collections import deque

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

    # Global JEPA Setup
    global_jepa = None
    jepa_optimizer = None
    replay_buffer = deque(maxlen=100000)
    
    if isinstance(policy, ParamNonLinearPolicy_JEPA):
        policy = cast(ParamNonLinearPolicy_JEPA, policy)
        # Use the config from the policy instance
        global_jepa = JEPAModule(
            input_dim=policy.n_features,
            hidden_dim=policy.jepa_hidden_dim,
            latent_dim=policy.jepa_latent_dim,
            action_dim=policy.n_actions,
        )
        jepa_optimizer = torch.optim.Adam(global_jepa.parameters(), lr=1e-3)

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
    global_best, best_params, global_best_gen = -np.inf, None, 0
    for gen in range(generations):
        ask_time = time.perf_counter()  # track time taken for evaluations
        X = es.ask()

        losses = []
        returns = []
        policy_infos = []
        population_traces = []
        
        # Collect all transitions for this generation
        gen_transitions = []

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
            ret, policy_info, transitions = evaluate_candidate(
                params=x,
                make_policy=make_policy,
                make_features=make_features,
                horizon=horizon,
                seed=seed,
                jepa=global_jepa # Pass global JEPA
            )
            returns.append(ret)
            losses.append(-ret)  # CMA-ES minimizes
            policy_infos.append(policy_info)
            gen_transitions.extend(transitions)

        es.tell(X, losses)
        elapsed = time.perf_counter() - ask_time  # time taken for evaluations
        ##es.disp()

        # Add to Replay Buffer
        #print(f"[Gen {gen:03d}] Transitions Generated: {len(gen_transitions)}")
        replay_buffer.extend(gen_transitions)
        #print(f"[Gen {gen:03d}] Replay Buffer Size: {len(replay_buffer)}")
        
        
        # Train Global JEPA
        if global_jepa is not None and len(replay_buffer) > 10000:
            # Mini-batch training
            batch_size = 256
            n_updates = 50 # Number of updates per generation
            
            jepa_losses = []
            
            for _ in range(n_updates):
                batch = random.sample(replay_buffer, batch_size)
                
                # Prepare batch tensors
                obs_batch = torch.tensor(np.stack([t["obs"] for t in batch])).float()
                next_obs_batch = torch.tensor(np.stack([t["next_obs"] for t in batch])).float()
                metadata_batch = torch.tensor(np.stack([t["metadata"] for t in batch])).float()
                
                # Action one-hot
                action_batch = torch.zeros(batch_size, policy.n_actions)
                actions = [t["action"] for t in batch]
                action_batch[range(batch_size), actions] = 1.0
                
                loss_dict = global_jepa.compute_losses(obs_batch, action_batch, next_obs_batch, metadata_batch)
                
                jepa_optimizer.zero_grad()
                loss_dict["loss"].backward()
                jepa_optimizer.step()
                global_jepa.update_target_encoder()
                
                jepa_losses.append(loss_dict["loss"].item())
            
            avg_jepa_loss = sum(jepa_losses) / len(jepa_losses)
            print(f"[Gen {gen:03d}] JEPA Loss: {avg_jepa_loss:.4f}")

        # vectorize all the candidate returns
        ret_array = np.array(returns, dtype=np.float32)
        gen_best = float(ret_array.max())
        gen_best_info = policy_infos[int(ret_array.argmax())]

        # Re-implementing trace logging briefly to match original file structure
        for idx, info in enumerate(policy_infos):
             actions = []
             # Limit trace size
             for step_idx, x in enumerate(info): 
                 action = int(x["selected_action"])
                 logits = x["logits"]
                 label = EndlessPlatformerEnv.ACTION_LABELS[action]
                 logits_list = (
                    logits.detach().cpu().numpy().tolist()
                    if hasattr(logits, "detach")
                    else getattr(logits, "tolist", lambda: list(logits))()
                 )
                 action_entry = {
                     "step": step_idx,
                     "action_index": action,
                     "action_label": label,
                     "logits": logits_list
                 }
                 if "curiosity_scores" in x:
                    action_entry["curiosity_scores"] = (
                        x["curiosity_scores"].detach().cpu().numpy().tolist()
                        if hasattr(x["curiosity_scores"], "detach")
                        else getattr(x["curiosity_scores"], "tolist", lambda: list(x["curiosity_scores"]))()
                    )
                 if "validity" in x:
                    try:
                        action_entry["validity"] = float(x["validity"])
                    except Exception:
                        action_entry["validity"] = (
                            x["validity"].detach().cpu().numpy().tolist()[0]
                            if hasattr(x["validity"], "detach")
                            else getattr(x["validity"], "tolist", lambda: list(x["validity"]))()[0]
                        )
                 actions.append(action_entry)
             population_traces.append({
                 "iteration": idx,
                 "return": float(returns[idx]),
                 "actions": actions
             })

        # find the global best policy parameters
        if gen_best > global_best:
            global_best = gen_best
            global_best_gen = gen
            best_params = X[
                int(ret_array.argmax())
            ].copy()  # store a copy so it's not overwritten

        stats = GenerationStats(
            generation=gen,
            wall_time_s=elapsed,
            best_reward=gen_best,
            # mean_reward=0,
            # median_reward=0,
            # reward_std=0,
            sigma=es.sigma,
            policy_info=gen_best_info,
            population_traces=population_traces,
        )
        history.append(stats)

        print(
            f"[Gen {gen:03d}] pop_mean={ret_array.mean():.2f} "
            f"pop_best={gen_best:.2f} global_best={global_best:.2f} (at gen {global_best_gen}) "
        )

        # For now, we run for a fixed number of generations without early stopping.
        # if es.stop():
        #     print("CMA-ES stopping criteria met. Ending training.")
        #     break

    # After training, create the best policy for return
    best_policy = make_policy()
    if best_params is not None:
        best_policy.set_params(best_params)
        
    # Inject the trained Global JEPA into the best policy
    if global_jepa is not None and isinstance(best_policy, ParamNonLinearPolicy_JEPA):
        best_policy.jepa = global_jepa

    # create a policy bundle to return
    return PolicyBundle(
        policy=best_policy,
        metadata={
            "best_gen": global_best_gen,
            "best_return": global_best,
            "dim": num_params,
            "popsize": lam,
        },
    ), history
