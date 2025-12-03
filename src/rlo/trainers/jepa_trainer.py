from __future__ import annotations

from math import e
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from rlo.envs.endless_platformer import EndlessPlatformerEnv
from rlo.policies.jepa import JEPAModule
from rlo.policies.param_base import Policy
from rlo.policies.param_nonlinear_jepa import ParamNonLinearPolicy_JEPA
from rlo.utils.logging import GenerationStats
from rlo.utils.serialization import PolicyBundle


class TransitionDataset(Dataset):
    """Simple dataset for (s, a, s') tuples collected from rollouts."""

    def __init__(self, transitions: list[Dict[str, Any]], n_actions: int):
        self.obs = torch.tensor(np.stack([t["obs"] for t in transitions]), dtype=torch.float32)
        self.next_obs = torch.tensor(np.stack([t["next_obs"] for t in transitions]), dtype=torch.float32)
        self.metadata = torch.tensor(np.stack([t["metadata"] for t in transitions]), dtype=torch.float32)

        actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long)
        self.action_one_hot = torch.zeros(len(transitions), n_actions, dtype=torch.float32)
        self.action_one_hot[torch.arange(len(transitions)), actions] = 1.0

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "obs": self.obs[idx],
            "next_obs": self.next_obs[idx],
            "action_one_hot": self.action_one_hot[idx],
            "metadata": self.metadata[idx],
        }


class JEPALightningModule(pl.LightningModule):
    """Lightning wrapper around :class:`JEPAModule`."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        tau: float = 0.995,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.jepa = JEPAModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            tau=tau,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor | float]:
        return self.jepa.compute_losses(
            obs=batch["obs"],
            action_one_hot=batch["action_one_hot"],
            next_obs=batch["next_obs"],
            next_metadata=batch["metadata"],
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss_dict = self.forward(batch)
        self.log("train_loss", loss_dict["loss"], prog_bar=True, on_epoch=True, on_step=False)
        # Float metrics are logged for monitoring/debugging.
        self.log("train_loss_trans", loss_dict["loss_trans"], on_epoch=True, on_step=False)
        self.log("train_loss_vicreg", float(loss_dict["loss_vicreg"]), on_epoch=True, on_step=False)
        self.log("train_loss_probe", loss_dict["loss_probe"], on_epoch=True, on_step=False)
        return loss_dict["loss"]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss_dict = self.forward(batch)
        self.log("val_loss", loss_dict["loss"], prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_loss_trans", loss_dict["loss_trans"], on_epoch=True, on_step=False)
        self.log("val_loss_vicreg", float(loss_dict["loss_vicreg"]), on_epoch=True, on_step=False)
        self.log("val_loss_probe", loss_dict["loss_probe"], on_epoch=True, on_step=False)
        return loss_dict["loss"]

    def on_train_batch_end(self, outputs: torch.Tensor | None, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        # Keep the target encoder synced with the online encoder.
        self.jepa.update_target_encoder()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def evaluate_candidate(
    make_policy: Callable[[], Policy],
    make_features: Callable[[Dict[str, Any], Dict[str, Any], int], np.ndarray],
    horizon: int = 1800,
    seed: int = 0,
) -> Tuple[float, list[Dict[str, np.ndarray]], list[Dict[str, object]]]:
    """Run one rollout to collect transitions for JEPA training."""

    # Create policy
    policy = make_policy()
    policy.reset()

    # Create environment
    env = EndlessPlatformerEnv(
        natural_drain_per_sec=0.1,
        move_drain_per_sec=0.15,
        jump_drain=0.2,
        seed=seed,
    )
    obs, info = env.reset(seed=seed)
    prev_action = -1
    total_reward = 0.0

    policy_info: list[Dict[str, Any]] = []
    transitions: list[Dict[str, Any]] = []

    for _ in range(horizon):
        features = make_features(obs, info, prev_action)
        action, act_info = policy.act(features, info)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        total_reward += float(reward)

        next_features = make_features(next_obs, next_info, action)
        energy = next_info.get("energy", 0.0)

        transitions.append(
            {
                "obs": features,
                "action": action,
                "next_obs": next_features,
                "metadata": np.array([energy], dtype=np.float32),
            }
        )

        policy_info.append(act_info)

        prev_action = action
        obs = next_obs
        info = next_info

        if terminated or truncated:
            break

    env.close()

    return total_reward, policy_info, transitions


def train_jepa(
    make_policy: Callable[[], Policy],
    make_features: Callable[[Dict[str, Any], Dict[str, Any], int], np.ndarray],
    base_seed: int = 42,
    horizon: int = 1800,
    *,
    num_episodes: int = 10,
    batch_size: int = 256,
    max_epochs: int = 10,
    val_split: float = 0.1,
    learning_rate: float = 1e-3,
    checkpoint_dir: str | Path = "runs/jepa",
    accelerator: str = "auto",
) -> list[GenerationStats]:
    """Train a global JEPA model using PyTorch Lightning and save the best checkpoint."""

    policy = make_policy()
    n_features, n_actions = policy.n_features, policy.n_actions

    returns = [] # should all be max horizon unless early termination
    history: list[GenerationStats] = []

    # Collect transitions from fresh rollouts.
    transitions: list[Dict[str, Any]] = []
    for episode in range(num_episodes):
        seed = base_seed + episode * 997
        ret, policy_info, episode_transitions = evaluate_candidate(
            make_policy=make_policy,
            make_features=make_features,
            horizon=horizon,
            seed=seed,
        )
        returns.append(ret)
        transitions.extend(episode_transitions)

        # vectorize all the candidate returns
        ret_array = np.array(returns, dtype=np.float32)
        gen_best = float(ret_array.max())

        # Re-implementing trace logging briefly to match original file structure
        population_traces = []
        actions = []

        for step_idx, x in enumerate(policy_info):
            action = int(x["selected_action"])
            label = EndlessPlatformerEnv.ACTION_LABELS[action]
            
            action_entry = {
                "step": step_idx,
                "action_index": action,
                "action_label": label,
            }
            
            actions.append(action_entry)
        
        population_traces.append({
            "iteration": episode,
            "return": float(ret),
            "actions": actions
        })

        
        stats = GenerationStats(
            generation=episode,
            wall_time_s=0,
            best_reward=gen_best,
            sigma=0,
            policy_info=policy_info,
            population_traces=population_traces,
        )
        history.append(stats)

    if not transitions:
        raise RuntimeError("No transitions collected for JEPA training.")
    else:
        print(f"Collected {len(transitions)} transitions for JEPA training.")

    dataset = TransitionDataset(transitions, n_actions=n_actions)
    val_size = int(len(dataset) * val_split) if val_split > 0 else 0
    val_size = min(max(val_size, 1 if len(dataset) > 1 else 0), len(dataset) - 1) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=15, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=15) if val_ds is not None else None

    model = JEPALightningModule(
        input_dim=n_features,
        hidden_dim=64,
        latent_dim=32,
        action_dim=n_actions,
        lr=learning_rate,
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monitor_metric = "val_loss" if val_loader is not None else "train_loss"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"jepa-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_checkpointing=True,
        default_root_dir=str(checkpoint_dir),
        accelerator=accelerator,
        devices=1,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_checkpoint = checkpoint_callback.best_model_path or None
    best_score = checkpoint_callback.best_model_score

    if best_checkpoint:
        best_module = JEPALightningModule.load_from_checkpoint(best_checkpoint)
        trained_jepa = best_module.jepa
    else:
        trained_jepa = model.jepa

    trained_jepa.eval()

    return history

    