"""Interactive viewer for the EndlessPlatformerEnv.

The player can:
    - Hold Right Arrow or 'd' to add a forward boost.
    - Tap/hold Up Arrow to jump.
    - Press Space to eat.
    - Press Escape or close the window to exit.

The viewer uses Tkinter (standard library) to display the grayscale frames at
the environment's native 160x90 resolution. Update the `PIXEL_SCALE` constant
to enlarge the window if desired.
"""

from __future__ import annotations

import argparse
import tkinter as tk
from typing import Dict

import numpy as np

from rlo.envs.endless_platformer import EndlessPlatformerEnv

PIXEL_SCALE = 3  # Scale factor for enlarging the viewport display.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play the Endless Platformer environment."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic layouts.",
    )
    return parser.parse_args()


class PlayerApp:
    """Minimal Tkinter-based viewer for the EndlessPlatformerEnv."""

    def __init__(self, env: EndlessPlatformerEnv, seed: int | None) -> None:
        self.env = env
        self.observation, _ = self.env.reset(seed=seed)
        self.root = tk.Tk()
        self.root.title("Endless Platformer Viewer")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        image = self.observation["image"]
        self.frame_height, self.frame_width = image.shape[:2]
        self.scale = PIXEL_SCALE
        self.canvas = tk.Canvas(
            self.root,
            width=self.frame_width * self.scale,
            height=self.frame_height * self.scale,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()

        self._base_photo = tk.PhotoImage(
            width=self.frame_width, height=self.frame_height
        )
        self._scaled_photo = self._base_photo.zoom(self.scale, self.scale)
        self._canvas_image = self.canvas.create_image(
            0, 0, anchor="nw", image=self._scaled_photo
        )

        self._keys_down: Dict[str, bool] = {}
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)

        self._color_cache = tuple(f"#{v:02x}{v:02x}{v:02x}" for v in range(256))
        self._running = True
        self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        frame_duration_ms = int(self.env.dt * 1000)
        self.root.after(frame_duration_ms, self._update)

    def _on_key_press(self, event: tk.Event) -> None:
        if event.keysym == "Escape":
            self.close()
        else:
            self._keys_down[event.keysym] = True

    def _on_key_release(self, event: tk.Event) -> None:
        self._keys_down.pop(event.keysym, None)

    def _resolve_action(self) -> int:
        if any(key in self._keys_down for key in ("Up",)):
            return self.env.JUMP
        if any(key in self._keys_down for key in ("Right",)):
            return self.env.FORWARD
        if any(key in self._keys_down for key in ("space",)):
            return self.env.EAT
        return self.env.NOOP

    def _update(self) -> None:
        if not self._running:
            return

        action = self._resolve_action()
        self.observation, _, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.observation, _ = self.env.reset()

        self._draw_frame(self.observation["image"][..., 0])
        self._schedule_next_frame()

    def _draw_frame(self, frame: np.ndarray) -> None:
        rows = [
            "{" + " ".join(self._color_cache[pixel] for pixel in row) + "}"
            for row in frame.astype(np.uint8)
        ]
        self._base_photo.put(" ".join(rows))
        # Refresh the scaled image reference to avoid Tkinter dropping it.
        self._scaled_photo = self._base_photo.zoom(self.scale, self.scale)
        self.canvas.itemconfig(self._canvas_image, image=self._scaled_photo)

    def close(self) -> None:
        self._running = False
        self.env.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    env = EndlessPlatformerEnv()
    app = PlayerApp(env, seed=args.seed)
    app.run()


if __name__ == "__main__":
    main()
