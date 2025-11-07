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
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Path to a saved policy bundle (.npz). If provided, the agent runs automatically.",
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
    return parser.parse_args()


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
        self._prob_var = tk.StringVar(value="")

        for var in (
            self._step_var,
            self._action_var,
            self._energy_var,
            self._state_var,
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
        flags = f"can_eat={bool(info_before.get('can_eat', False))}  on_ground={bool(info_after.get('on_ground', False))}"
        self._state_var.set(f"Flags: {flags}")

        probs = diagnostics["probabilities"]
        prob_lines = []
        for idx, env_action in enumerate(self._action_labels.keys()):
            label = self._action_labels.get(env_action, str(env_action)).upper()
            prob_lines.append(f"{label:>8}: {probs[idx]:5.2f}")
        self._prob_var.set("\n".join(prob_lines))

        features = diagnostics["features"]
        contributions = diagnostics["contributions"][diagnostics["local_index"]]
        aero = list(zip(FEATURE_NAMES, features, contributions))
        aero.sort(key=lambda item: abs(item[2]), reverse=True)
        lines = [
            f"{name:>22}: {value:+.3f}  logit={impact:+.3f}"
            for name, value, impact in aero[:8]
        ]
        self._features_label.config(text="\n".join(lines))


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
    ) -> None:
        self.seed = seed
        self.env = env
        self.policy_controller = (
            PolicyController(policy_bundle, deterministic=deterministic_policy)
            if policy_bundle is not None
            else None
        )
        self.show_scratchpad = show_scratchpad and self.policy_controller is not None
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
        if self.show_scratchpad and self.policy_controller:
            self.scratchpad = ScratchpadPanel(
                self._container, self.policy_controller.bundle, self.env.ACTION_LABELS
            )
            self.scratchpad.frame.pack(side="right", fill="y", padx=8, pady=8)

        self._keys_down: Dict[str, bool] = {}
        if self.policy_controller is None:
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

        if self.policy_controller is not None:
            action, diagnostics = self.policy_controller.act(current_obs, current_info)
        else:
            action = self._resolve_manual_action()
            diagnostics = None

        self.action = action
        next_obs, reward, terminated, truncated, next_info = self.env.step(action)
        self.observation, self.info = next_obs, next_info
        self._step_counter += 1

        if terminated or truncated:
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
    if args.policy is not None:
        bundle = PolicyBundle.load(args.policy)
        print(f"Loaded policy bundle from {args.policy}")
        if not args.scratchpad:
            print("Tip: re-run with --scratchpad to see the policy diagnostics panel.")

    env = EndlessPlatformerEnv(
        natural_drain_per_sec=0.1,
        move_drain_per_sec=0.1,
        jump_drain=0.25,
        render_mode="rgb_array",
        seed=args.seed,
    )
    app = PlayerApp(
        env,
        seed=args.seed,
        policy_bundle=bundle,
        deterministic_policy=not args.stochastic,
        show_scratchpad=args.scratchpad,
        pixel_scale=args.scale,
    )
    app.run()


if __name__ == "__main__":
    main()
