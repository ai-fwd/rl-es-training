"""Interactive viewer for the EndlessPlatformerEnv with optional policy playback.

This tool serves two audiences:

1. Manual control for quick environment inspection using the keyboard.
2. Visualising trained CMA-ES policies with a live "scratchpad" that explains
   which features drove each action choice.

Keyboard controls (manual mode):
    - Hold Right Arrow or 'd' to move forward.
    - Press Space to eat an orb when intersecting it.
    - Press Up Arrow to jump (not used by the CMA-ES policy but kept for humans).
    - Press Escape or close the window to exit.

When a policy bundle is supplied (``--policy``) the app switches to autoplay.
Enable ``--scratchpad`` to reveal detailed diagnostics alongside the viewport.
"""

from __future__ import annotations

import argparse
import json
import pickle
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rlo.envs import EndlessPlatformerEnv
from rlo.features import FEATURE_NAMES, make_basic_features
from rlo.utils.serialization import PolicyBundle

PIXEL_SCALE_DEFAULT = 3  # Scale factor for enlarging the viewport display.


# --------------------------------------------------------------------------- #
# Command-line parsing                                                        #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise the Endless Platformer environment and policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic layouts.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Path to a saved policy bundle (.npz). If provided, the agent runs automatically.",
    )
    mode_group.add_argument(
        "--replay-file",
        type=Path,
        default=None,
        help="Path to a history.json or transitions-*.pkl produced by JEPA training. Plays back recorded actions without a policy.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of greedily picking the maximum probability option.",
    )
    parser.add_argument(
        "--scratchpad",
        action="store_true",
        help="Show a diagnostics panel explaining the policy's action selection.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=PIXEL_SCALE_DEFAULT,
        help="Scaling factor applied to the viewport.",
    )
    parser.add_argument(
        "--simulation-speed",
        type=float,
        default=1.0,
        help="Multiplier for the simulation speed (1.0 = real-time).",
    )
    parser.add_argument(
        "--replay-generation",
        type=int,
        default=0,
        help="Generation index to replay from the history file.",
    )
    parser.add_argument(
        "--replay-iteration",
        type=int,
        default=0,
        help="Iteration within the generation's population traces to replay.",
    )
    parser.add_argument(
        "--policy-args",
        nargs="*",
        help="Optional key=value overrides for policy parameters (e.g. temperature=0.5).",
    )
    return parser.parse_args()


@dataclass
class ReplayTrace:
    actions: List[int]
    generation: Optional[int] = None
    iteration: Optional[int] = None
    reward: Optional[float] = None


def _load_history_from_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict) and "history" in payload:
        return payload["history"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unrecognised JSON format in {path}")


def _load_history_from_pickle(path: Path) -> List[Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict) and "history" in payload:
        return payload["history"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unrecognised pickle format in {path}")


def _get_generation(entry: Any) -> Optional[int]:
    if isinstance(entry, dict):
        return entry.get("generation")
    return getattr(entry, "generation", None)


def _get_population_traces(entry: Any) -> List[Dict[str, Any]]:
    if isinstance(entry, dict):
        return entry.get("population") or entry.get("population_traces") or []
    return getattr(entry, "population_traces", []) or []


def _get_candidate_iteration(candidate: Any) -> Optional[int]:
    if isinstance(candidate, dict):
        return candidate.get("iteration")
    return getattr(candidate, "iteration", None)


def _get_actions_from_candidate(candidate: Any) -> List[Any]:
    if isinstance(candidate, dict):
        return candidate.get("actions") or candidate.get("trace") or []
    return getattr(candidate, "actions", []) or []


def _normalize_actions(action_entries: List[Any]) -> List[int]:
    actions: List[int] = []
    for entry in action_entries:
        if isinstance(entry, dict):
            if "action_index" in entry:
                actions.append(int(entry["action_index"]))
            elif "action" in entry:
                actions.append(int(entry["action"]))
            elif "selected_action" in entry:
                actions.append(int(entry["selected_action"]))
        else:
            try:
                actions.append(int(entry))
            except Exception:
                continue
    return actions


def _get_best_trace(entry: Any) -> List[Any]:
    if isinstance(entry, dict):
        best_policy = entry.get("best_policy") or {}
        return best_policy.get("trace") or []
    policy_info = getattr(entry, "policy_info", None)
    if isinstance(policy_info, list):
        return policy_info
    return []


def _get_candidate_return(candidate: Any, fallback_best: Optional[float]) -> Optional[float]:
    if isinstance(candidate, dict):
        if "return" in candidate:
            return float(candidate["return"])
    else:
        val = getattr(candidate, "return_", None) or getattr(candidate, "return_val", None)
        if val is not None:
            try:
                return float(val)
            except Exception:
                pass
    return fallback_best


def load_replay_trace(path: Path, generation: int, iteration: int) -> ReplayTrace:
    if not path.exists():
        raise FileNotFoundError(f"Replay file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        history = _load_history_from_json(path)
    elif suffix == ".pkl":
        history = _load_history_from_pickle(path)
    else:
        raise ValueError("Replay file must be a .json or .pkl produced by JEPA training.")

    target = None
    for entry in history:
        if _get_generation(entry) == generation:
            target = entry
            break

    if target is None:
        raise ValueError(f"Generation {generation} not found in {path}")

    population = _get_population_traces(target)
    candidate = next(
        (cand for cand in population if _get_candidate_iteration(cand) == iteration),
        None,
    )
    if candidate is None and iteration < len(population):
        candidate = population[iteration]

    actions: List[int] = []
    candidate_return: Optional[float] = None
    best_reward: Optional[float] = None
    if isinstance(target, dict):
        best_reward = target.get("best_reward")
    else:
        best_reward = getattr(target, "best_reward", None)

    if candidate is not None:
        actions = _normalize_actions(_get_actions_from_candidate(candidate))
        candidate_return = _get_candidate_return(candidate, best_reward)

    if not actions:
        actions = _normalize_actions(_get_best_trace(target))
        if candidate_return is None:
            candidate_return = best_reward

    if not actions:
        raise ValueError(
            f"No actions found for generation={generation}, iteration={iteration} in {path}"
        )

    return ReplayTrace(
        actions=actions,
        generation=generation,
        iteration=iteration,
        reward=candidate_return,
    )


# --------------------------------------------------------------------------- #
# Policy controller & diagnostics                                             #
# --------------------------------------------------------------------------- #


@dataclass
class PolicyController:
    """Bridge between a :class:`PolicyBundle` and the environment loop."""

    bundle: PolicyBundle
    deterministic: bool

    def __post_init__(self) -> None:
        self.extractor = make_basic_features
        self._prev_action = 0  # Start with NOOP

    def reset(self) -> None:
        self._prev_action = 0

    def act(self, observation: Dict, info: Dict) -> Tuple[int, Dict]:
        """Return an environment action id and diagnostic details."""
        features = self.extractor(observation, info, self._prev_action)
        action_id, diagnostics = self.bundle.policy.act(features, info)
        self._prev_action = action_id
        diagnostics = dict(diagnostics)
        diagnostics["features"] = features
        return action_id, diagnostics


@dataclass
class ReplayController:
    """Simple cursor over a pre-recorded action trace."""

    trace: ReplayTrace

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._cursor = 0

    def next_action(self) -> int:
        if self._cursor >= len(self.trace.actions):
            raise StopIteration
        action = int(self.trace.actions[self._cursor])
        self._cursor += 1
        return action

    def finished(self) -> bool:
        return self._cursor >= len(self.trace.actions)

class ReplayScratchpad:
    """Minimal scratchpad for replay mode."""

    def __init__(self, parent: tk.Misc, trace: ReplayTrace, action_labels: Dict[int, str]) -> None:
        self._frame = tk.Frame(parent, padx=10, pady=10, bg="#1e1e1e")
        self._frame.configure(highlightbackground="#444", highlightthickness=1)

        title = tk.Label(
            self._frame,
            text="Replay Info",
            font=("TkDefaultFont", 14, "bold"),
            fg="#ffffff",
            bg="#1e1e1e",
        )
        title.pack(anchor="w")

        reward_text = (
            f"Max reward: {trace.reward:.3f}"
            if trace.reward is not None
            else "Max reward: —"
        )
        self._reward_var = tk.StringVar(value=reward_text)
        self._step_var = tk.StringVar(value="Step: 0")
        self._action_var = tk.StringVar(value="Action: —")

        for var in (self._reward_var, self._step_var, self._action_var):
            tk.Label(self._frame, textvariable=var, fg="#d7d7d7", bg="#1e1e1e").pack(
                anchor="w"
            )

        self._action_labels = action_labels

    @property
    def frame(self) -> tk.Frame:
        return self._frame

    def update(self, step: int, action_id: int) -> None:
        action_label = self._action_labels.get(action_id, f"id={action_id}")
        self._step_var.set(f"Step: {step}")
        self._action_var.set(f"Action: {action_label.upper()}")

class ScratchpadPanel:
    """Tkinter widget that renders policy insights alongside the viewport."""

    def __init__(
        self, parent: tk.Misc, bundle: PolicyBundle, action_labels: Dict[int, str]
    ) -> None:
        self._bundle = bundle
        self._action_labels = action_labels
        self._frame = tk.Frame(parent, padx=10, pady=10, bg="#1e1e1e")
        self._frame.configure(highlightbackground="#444", highlightthickness=1)

        title = tk.Label(
            self._frame,
            text="Policy Scratchpad",
            font=("TkDefaultFont", 14, "bold"),
            fg="#ffffff",
            bg="#1e1e1e",
        )
        title.pack(anchor="w")

        self._step_var = tk.StringVar(value="Step: 0")
        self._action_var = tk.StringVar(value="Action: —")
        self._energy_var = tk.StringVar(value="Energy: —")
        self._state_var = tk.StringVar(value="Flags: —")
        self._validity_var = tk.StringVar(value="")
        self._prob_var = tk.StringVar(value="")

        for var in (
            self._step_var,
            self._action_var,
            self._energy_var,
            self._state_var,
            self._validity_var,
        ):
            tk.Label(self._frame, textvariable=var, fg="#d7d7d7", bg="#1e1e1e").pack(
                anchor="w"
            )

        tk.Label(
            self._frame,
            text="Action probabilities:",
            fg="#ffffff",
            bg="#1e1e1e",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", pady=(12, 2))
        tk.Label(
            self._frame,
            textvariable=self._prob_var,
            fg="#d7d7d7",
            bg="#1e1e1e",
            justify="left",
        ).pack(anchor="w")

        tk.Label(
            self._frame,
            text="Top feature contributions:",
            fg="#ffffff",
            bg="#1e1e1e",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", pady=(12, 2))

        self._features_label = tk.Label(
            self._frame,
            fg="#d7d7d7",
            bg="#1e1e1e",
            justify="left",
            font=("Courier", 10),
        )
        self._features_label.pack(anchor="w")

    @property
    def frame(self) -> tk.Frame:
        return self._frame

    def clear(self) -> None:
        self._step_var.set("Step: 0")
        self._action_var.set("Action: —")
        self._energy_var.set("Energy: —")
        self._state_var.set("Flags: —")
        self._validity_var.set("")
        self._prob_var.set("")
        self._features_label.config(text="")

    def update(
        self,
        step: int,
        action_id: int,
        diagnostics: Dict,
        metrics: np.ndarray,
        info_before: Dict,
        info_after: Dict,
    ) -> None:
        action_label = self._action_labels.get(action_id, f"id={action_id}")
        self._step_var.set(f"Step: {step}")
        self._action_var.set(f"Action: {action_label.upper()}")
        energy = float(metrics[0])
        self._energy_var.set(f"Energy: {energy:.3f}")
        flags = f"can_eat={bool(info_before.get('can_eat', False))} near_food={bool(info_before.get('near_food', False))} on_ground={bool(info_after.get('on_ground', False))}"
        self._state_var.set(f"Flags: {flags}")

        if "validity" in diagnostics:
             self._validity_var.set(f"Validity: {diagnostics['validity']:.3f}")
        else:
             self._validity_var.set("")

        if "probabilities" in diagnostics:
            probs = diagnostics["probabilities"]
        elif "logits" in diagnostics:
            # Softmax
            logits = diagnostics["logits"]
            if hasattr(logits, "detach"):
                logits = logits.detach().cpu().numpy()
            if isinstance(logits, list):
                logits = np.array(logits)
            
            # Handle batch dim if present
            if logits.ndim > 1:
                logits = logits.flatten()
                
            e_x = np.exp(logits - np.max(logits))
            probs = e_x / e_x.sum()
        else:
            probs = [0.0] * len(self._action_labels)

        prob_lines = []
        for idx, env_action in enumerate(self._action_labels.keys()):
            label = self._action_labels.get(env_action, str(env_action)).upper()
            val = probs[idx] if idx < len(probs) else 0.0
            prob_lines.append(f"{label:>8}: {val:5.2f}")
        self._prob_var.set("\n".join(prob_lines))

        if "curiosity_scores" in diagnostics:
            scores = diagnostics["curiosity_scores"]
            lines = ["Curiosity Scores:"]
            for idx, env_action in enumerate(self._action_labels.keys()):
                label = self._action_labels.get(env_action, str(env_action)).upper()
                val = scores[idx] if idx < len(scores) else 0.0
                lines.append(f"{label:>8}: {val:+.3f}")
            self._features_label.config(text="\n".join(lines))
        elif "contributions" in diagnostics:
            features = diagnostics["features"]
            contributions = diagnostics["contributions"][diagnostics["selected_action"]]
            aero = list(zip(FEATURE_NAMES, features, contributions))
            aero.sort(key=lambda item: abs(item[2]), reverse=True)
            lines = [
                f"{name:>22}: {value:+.3f}  logit={impact:+.3f}"
                for name, value, impact in aero[:8]
            ]
            self._features_label.config(text="\n".join(lines))
        else:
            self._features_label.config(text="No feature attribution available")


class InfoScratchpad:
    """Dynamic scratchpad that visualizes the environment info dictionary."""

    def __init__(self, parent: tk.Misc) -> None:
        self._frame = tk.Frame(parent, padx=10, pady=10, bg="#1e1e1e")
        self._frame.configure(highlightbackground="#444", highlightthickness=1)

        title = tk.Label(
            self._frame,
            text="Environment Info",
            font=("TkDefaultFont", 14, "bold"),
            fg="#ffffff",
            bg="#1e1e1e",
        )
        title.pack(anchor="w", pady=(0, 10))

        self._vars: Dict[str, tk.StringVar] = {}
        self._labels: Dict[str, tk.Label] = {}

    @property
    def frame(self) -> tk.Frame:
        return self._frame

    def clear(self) -> None:
        # No-op or reset all vars?
        pass

    def update(self, info: Dict[str, Any]) -> None:
        seen_keys = set()
        
        # Sort keys for stable display order
        for key in sorted(info.keys()):
            val = info[key]
            # Simple formatting
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            
            display_text = f"{key}: {val_str}"
            seen_keys.add(key)

            if key not in self._vars:
                # Create new entry
                var = tk.StringVar(value=display_text)
                label = tk.Label(
                    self._frame,
                    textvariable=var,
                    fg="#d7d7d7",
                    bg="#1e1e1e",
                    anchor="w",
                    justify="left"
                )
                label.pack(fill="x", anchor="w")
                self._vars[key] = var
                self._labels[key] = label
            else:
                # Update existing entry
                self._vars[key].set(display_text)
                # Ensure it's packed (in case it was hidden, though we destroy hidden ones below)
                if not self._labels[key].winfo_ismapped():
                     self._labels[key].pack(fill="x", anchor="w")

        # Cleanup keys that are no longer present
        current_keys = list(self._labels.keys())
        for key in current_keys:
            if key not in seen_keys:
                self._labels[key].destroy()
                del self._labels[key]
                del self._vars[key]



# --------------------------------------------------------------------------- #
# Player application                                                          #
# --------------------------------------------------------------------------- #


class PlayerApp:
    """Minimal Tkinter-based viewer for the EndlessPlatformerEnv."""

    def __init__(
        self,
        env: EndlessPlatformerEnv,
        seed: Optional[int],
        *,
        policy_bundle: Optional[PolicyBundle] = None,
        deterministic_policy: bool = True,
        show_scratchpad: bool = False,
        pixel_scale: int = PIXEL_SCALE_DEFAULT,
        replay_trace: Optional[ReplayTrace] = None,
    ) -> None:
        self.seed = seed
        self.env = env
        self.policy_controller = (
            PolicyController(policy_bundle, deterministic=deterministic_policy)
            if policy_bundle is not None
            else None
        )
        self.replay_controller = (
            ReplayController(replay_trace) if replay_trace is not None else None
        )
        self.replay_trace = replay_trace
        self.show_scratchpad = show_scratchpad
        self.pixel_scale = max(1, int(pixel_scale))


        self.root = tk.Tk()
        self.root.title("Endless Platformer Viewer")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        # Layout: left = viewport, right = scratchpad (optional).
        self._container = tk.Frame(self.root, bg="#1b1b1b")
        self._container.pack(fill="both", expand=True)

        self.observation, self.info = self.env.reset(seed=self.seed)
        if self.policy_controller:
            self.policy_controller.reset()

        # Render to determine dimensions.
        init_rgb = self.env.render()
        self.frame_height, self.frame_width = init_rgb.shape[:2]
        self.canvas = tk.Canvas(
            self._container,
            width=self.frame_width * self.pixel_scale,
            height=self.frame_height * self.pixel_scale,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(side="left", fill="both", expand=False)

        self._base_photo = tk.PhotoImage(
            width=self.frame_width, height=self.frame_height
        )
        self._scaled_photo = self._base_photo.zoom(self.pixel_scale, self.pixel_scale)
        self._canvas_image = self.canvas.create_image(
            0, 0, anchor="nw", image=self._scaled_photo
        )

        self.scratchpad = None
        if self.show_scratchpad:
            if self.policy_controller:
                self.scratchpad = ScratchpadPanel(
                    self._container, self.policy_controller.bundle, self.env.ACTION_LABELS
                )
            elif self.replay_controller and self.replay_trace:
                self.scratchpad = ReplayScratchpad(
                    self._container, self.replay_trace, self.env.ACTION_LABELS
                )
            else:
                self.scratchpad = InfoScratchpad(self._container)
            
            self.scratchpad.frame.pack(side="right", fill="y", padx=8, pady=8)
            


        self._keys_down: Dict[str, bool] = {}
        self._manual_control = (
            self.policy_controller is None and self.replay_controller is None
        )
        if self._manual_control:
            self.root.bind("<KeyPress>", self._on_key_press)
            self.root.bind("<KeyRelease>", self._on_key_release)
        else:
            # Still honour Escape to exit autoplay.
            self.root.bind("<KeyPress-Escape>", lambda _event: self.close())

        self._running = True
        self._step_counter = 0
        self._schedule_next_frame()

    # --- Tk callbacks --------------------------------------------------- #
    def _schedule_next_frame(self) -> None:
        frame_duration_ms = max(1, int(self.env.dt * 1000))
        self.root.after(frame_duration_ms, self._update)

    def _on_key_press(self, event: tk.Event) -> None:
        if event.keysym == "Escape":
            self.close()
        else:
            self._keys_down[event.keysym] = True

    def _on_key_release(self, event: tk.Event) -> None:
        self._keys_down.pop(event.keysym, None)

    def _resolve_manual_action(self) -> int:
        if "Up" in self._keys_down:
            return self.env.JUMP
        if any(key in self._keys_down for key in ("Right", "d")):
            return self.env.FORWARD
        if "space" in self._keys_down:
            return self.env.EAT
        return self.env.NOOP

    # --- Main loop ------------------------------------------------------ #
    def _update(self) -> None:
        if not self._running:
            return

        current_obs = self.observation
        current_info = self.info
        current_metrics = current_obs["metrics"]

        if self.replay_controller is not None:
            if self.replay_controller.finished():
                print("Replay finished.")
                self.close()
                return
            try:
                action = self.replay_controller.next_action()
            except StopIteration:
                print("Replay finished.")
                self.close()
                return
            diagnostics = None
        elif self.policy_controller is not None:
            action, diagnostics = self.policy_controller.act(current_obs, current_info)
        else:
            action = self._resolve_manual_action()
            diagnostics = None

        self.action = action
        next_obs, reward, terminated, truncated, next_info = self.env.step(action)
        self.observation, self.info = next_obs, next_info
        self._step_counter += 1

        if terminated or truncated:
            if self.replay_controller is not None:
                print("Episode ended during replay.")
                self.close()
                return
            self.observation, self.info = self.env.reset(seed=self.seed)
            if self.policy_controller:
                self.policy_controller.reset()
            if self.scratchpad:
                self.scratchpad.clear()
            self._step_counter = 0

        if self.scratchpad and diagnostics is not None:
            self.scratchpad.update(
                step=self._step_counter,
                action_id=action,
                diagnostics=diagnostics,
                metrics=current_metrics,
                info_before=current_info,
                info_after=next_info,
            )
        if self.scratchpad and diagnostics is None and self.replay_controller:
            self.scratchpad.update(
                step=self._step_counter,
                action_id=action,
            )
        else:
            self.scratchpad.update(next_info)


        rgb = self.env.render()
        self._draw_frame(rgb)
        self._schedule_next_frame()

    # --- Rendering ------------------------------------------------------ #
    def _draw_frame(self, frame: np.ndarray) -> None:
        h, w, _ = frame.shape
        rows = []
        for y in range(h):
            row = frame[y]
            parts = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in row]
            rows.append("{" + " ".join(parts) + "}")
        self._base_photo.put(" ".join(rows))
        self._scaled_photo = self._base_photo.zoom(self.pixel_scale, self.pixel_scale)
        self.canvas.itemconfig(self._canvas_image, image=self._scaled_photo)

    # --- Lifecycle ------------------------------------------------------ #
    def close(self) -> None:
        self._running = False
        self.env.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# --------------------------------------------------------------------------- #
# Entrypoint                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()

    bundle: Optional[PolicyBundle] = None
    replay_trace: Optional[ReplayTrace] = None
    if args.replay_file is not None:
        replay_trace = load_replay_trace(
            args.replay_file, args.replay_generation, args.replay_iteration
        )
        print(
            f"Loaded {len(replay_trace.actions)} recorded actions from {args.replay_file} "
            f"(generation={args.replay_generation}, iteration={args.replay_iteration})"
        )
        if replay_trace.reward is not None:
            print(f"Recorded reward for iteration: {replay_trace.reward:.3f}")
        if args.scratchpad:
            print("Scratchpad limited to replay metadata (no policy diagnostics available).")
    elif args.policy is not None:
        # Initialize ParamReader
        from rlo.params import ParamReader
        reader = ParamReader.get_instance()
        
        # 1. Try to load params.yaml from the policy directory
        policy_path = Path(args.policy)
        local_params = policy_path.parent / "params.yaml"
        if local_params.exists():
            print(f"Loading config from {local_params}")
            reader.load(str(local_params))
        else:
            # Fallback to project root params.yaml if exists
            # Assuming script is run from project root or we can find it
            default_params = Path("params.yaml")
            if default_params.exists():
                print(f"Loading config from {default_params}")
                reader.load(str(default_params))

        # 2. Parse overrides
        overrides = {}
        if args.policy_args:
            for item in args.policy_args:
                if "=" not in item:
                    print(f"Warning: ignoring malformed policy arg '{item}' (expected key=value)")
                    continue
                key, val_str = item.split("=", 1)
                # Simple type inference
                if val_str.lower() == "true":
                    val = True
                elif val_str.lower() == "false":
                    val = False
                else:
                    try:
                        val = int(val_str)
                    except ValueError:
                        try:
                            val = float(val_str)
                        except ValueError:
                            val = val_str
                overrides[key] = val
            print(f"Policy overrides: {overrides}")
            reader.set_overrides(**overrides)

        # We don't pass overrides to PolicyBundle.load anymore, 
        # as the policy will query ParamReader.
        bundle = PolicyBundle.load(args.policy)
        print(f"Loaded policy bundle from {args.policy}")
        if not args.scratchpad:
            print("Tip: re-run with --scratchpad to see the policy diagnostics panel.")

    env = EndlessPlatformerEnv(
        render_mode="rgb_array",
        seed=args.seed,
        simulation_speed=args.simulation_speed,
        is_training=False,
        # natural_drain_per_sec=0.1,
        # move_drain_per_sec=0.15,
        # jump_drain=0.2,
    )
    app = PlayerApp(
        env,
        seed=args.seed,
        policy_bundle=bundle,
        deterministic_policy=not args.stochastic,
        show_scratchpad=args.scratchpad,
        pixel_scale=args.scale,
        replay_trace=replay_trace,
    )

    app.run()


if __name__ == "__main__":
    main()
