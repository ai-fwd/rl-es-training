"""Endless platform walker environment compatible with Gymnasium.

The environment models a side-scrolling character that advances over a
platform with fixed-width gaps that appear at random intervals. The agent
receives grayscale pixel observations of the current viewport and can choose
between three discrete actions:

    0 - NOOP: no movement
    1 - FORWARD: move forward
    2 - JUMP: initiate a jump when on the platform (also grants forward burst)
    3 - EAT

The simulation runs at 60 FPS, never terminates on its own, and respawns the
character on the last safe ground tile after a fall, enabling continuous rollouts.
"""

from __future__ import annotations

import importlib.resources as pkg_resources
from typing import Dict, List, Tuple

import numpy as np
from gymnasium import Env, spaces
from gymnasium.utils import seeding


class EndlessPlatformerEnv(Env):
    """Side-scrolling platform environment with an endless stream of gaps."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    NOOP = 0
    FORWARD = 1
    JUMP = 2
    EAT = 3

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        gap_width: float = 2.2,
        gap_probability: float = 0,
        segment_length: float = 8.0,
        # Energy mechanics (tunable)
        natural_drain_per_sec: float = 0.01,
        move_drain_per_sec: float = 0.01,
        jump_drain: float = 0.05,
        energy_pickup_amount: float = 0.25,
        energy_max_overfill: float = 1.5,
        seed: int | None = None,
    ) -> None:
        """Initialise the environment.

        Args:
            render_mode: Optional Gymnasium render mode (supports "rgb_array").
            gap_width: Fixed physical width of every platform break.
            gap_probability: Probability that a new segment contains a gap.
            segment_length: Length of each procedurally generated segment.
            seed: Optional RNG seed for reproducibility.
        """
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self.np_random = None

        # Simulation timing and physics constants.
        self.dt = 1.0 / 60.0
        self.gravity = -28.0
        self.jump_velocity = 8.0
        self.base_speed = 3.0
        self.forward_boost = 2.5

        # Character dimensions and placement.
        self.character_width = 1.4
        self.character_height = 1.4
        self.ground_height = 0.0
        self.platform_thickness = 0.5
        self.fall_threshold = -1.0

        # Procedural generation parameters.
        self.gap_width = gap_width
        self.gap_probability = gap_probability
        self.segment_length = segment_length
        self.segment_margin = 1.5
        self.initial_safe_span = 12.0
        self.cleanup_margin = 5.0

        # Energy system configuration
        self.natural_drain_per_sec = float(natural_drain_per_sec)
        self.move_drain_per_sec = float(move_drain_per_sec)
        self.jump_drain = float(jump_drain)
        self.energy_pickup_amount = float(energy_pickup_amount)
        self.energy_max_overfill = float(energy_max_overfill)

        # Viewport configuration for rendering.
        self.frame_width = 160
        self.frame_height = 90
        self.view_backward = 3.5
        self.view_forward = 6.5
        self.view_height = 8.0

        # Observations: Dict with image and metrics [energy, max_energy, elapsed_time_s]
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.frame_height, self.frame_width, 1),
                    dtype=np.uint8,
                ),
                "metrics": spaces.Box(
                    low=np.array([0.0, 1.0, 0.0], dtype=np.float32),
                    high=np.array(
                        [self.energy_max_overfill, self.energy_max_overfill, 1e9],
                        dtype=np.float32,
                    ),
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(4)

        self._character_x = 0.0
        self._character_y = 0.0
        self._character_vy = 0.0
        self._on_ground = True
        self._last_safe_x = 0.0

        self._gap_positions: List[Tuple[float, float]] = []
        self._generated_until = 0.0

        # Energy orbs and energy/time state
        self._energy_orbs: List[Tuple[float, float, float]] = []
        self._elapsed_time = 0.0
        self._energy = 1.0

        self._last_frame: np.ndarray | None = None
        # Lazy-loaded sprite cache (rgb uint8) and transparency key
        self._sprite_rgb: np.ndarray | None = None
        self._sprite_bg_rgb: Tuple[int, int, int] | None = None

    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict | None = None,
    ) -> Tuple[Dict, Dict]:
        """Reset the environment state."""
        del options
        if seed is not None:
            self._seed = seed
        if self.np_random is None or seed is not None:
            self.np_random, actual_seed = seeding.np_random(self._seed)
            self._seed = actual_seed

        self._character_x = 1.0
        self._character_y = self.ground_height
        self._character_vy = 0.0
        self._on_ground = True
        self._last_safe_x = self._character_x

        self._gap_positions = []
        self._generated_until = 0.0
        self._energy_orbs = []
        self._elapsed_time = 0.0
        self._energy = 1.0
        self._generate_platform_until(self._character_x + self.view_forward + 10.0)

        self._last_frame = None
        observation = {
            "image": self._render_observation(),
            "metrics": self._get_metrics_vector(),
        }
        info = {
            "x_position": self._character_x,
            "y_position": self._character_y,
            "on_ground": self._on_ground,
            "gaps": tuple(self._gap_positions),
            "energy": self._energy,
            "energy_max": self.energy_max_overfill,
            "elapsed_time_s": self._elapsed_time,
        }
        return observation, info

    def step(
        self,
        action: int,
    ) -> Tuple[Dict, float, bool, bool, Dict]:
        """Advance the simulation by one frame."""
        action = int(action)
        assert self.action_space.contains(action), f"{action} is not a valid action."

        forward_speed = self.base_speed
        forward_boost = self.forward_boost
        jump_velocity = self.jump_velocity

        # if energy is at max then you move slower until back under 1.0
        if self._energy >= 1.0:
            forward_speed *= 0.5
            forward_boost *= 0.5
            jump_velocity *= 0.5

        # Apply forward boost when jumping
        if action == self.JUMP:
            forward_speed += forward_boost

        # Horizontal movement only when forward action is taken
        previous_x = self._character_x
        if action == self.FORWARD or action == self.JUMP:
            self._character_x += forward_speed * self.dt

        # Jump initiation.
        if action == self.JUMP and self._on_ground:
            self._character_vy = jump_velocity
            self._on_ground = False
            # One-shot jump drain
            self._energy -= self.jump_drain

        # Apply gravity.
        self._character_vy += self.gravity * self.dt
        self._character_y += self._character_vy * self.dt

        # Time and continuous drains
        self._elapsed_time += self.dt
        self._energy -= self.natural_drain_per_sec * self.dt
        if action == self.FORWARD:
            self._energy -= self.move_drain_per_sec * self.dt
        if self._energy < 0.0:
            self._energy = 0.0

        support = self._has_support()
        fell = False
        if support and self._character_y <= self.ground_height:
            self._character_y = self.ground_height
            self._character_vy = 0.0
            self._on_ground = True
            self._last_safe_x = self._character_x
        else:
            self._on_ground = False

        if self._character_y < self.fall_threshold:
            # Respawn on the last safe tile to keep the rollout alive.
            self._character_x = self._last_safe_x
            self._character_y = self.ground_height
            self._character_vy = 0.0
            self._on_ground = True
            fell = True

        self._generate_platform_until(self._character_x + self.view_forward + 10.0)
        self._prune_old_gaps(
            self._character_x - self.view_backward - self.cleanup_margin
        )

        # if collision with energy orb
        orb = self._handle_energy_orbs_collision()
        if orb is not None:
            if action == self.EAT and self._energy < 1.0:
                # Consume
                self._energy = min(
                    self.energy_max_overfill, self._energy + self.energy_pickup_amount
                )
                self._energy_orbs.remove(orb)

        observation = {
            "image": self._render_observation(),
            "metrics": self._get_metrics_vector(),
        }
        # Constant reward per step until energy depletes to mimic survival-only reward
        reward = 1.0
        terminated = self._energy <= 0.0
        truncated = False
        info = {
            "x_position": self._character_x,
            "y_position": self._character_y,
            "x_velocity": forward_speed,
            "y_velocity": self._character_vy,
            "on_ground": self._on_ground,
            "fell": fell,
            "can_eat": orb is not None,
            "energy": self._energy,
            "energy_max": self.energy_max_overfill,
            "elapsed_time_s": self._elapsed_time,
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Return an RGB frame when render_mode is 'rgb_array'."""
        if self.render_mode != "rgb_array":
            raise ValueError("Only the 'rgb_array' render mode is supported.")
        # Produce a color frame for visualization without affecting observation logic
        return self._render_rgb_frame()

    def close(self) -> None:
        """Clean up any renderer state."""
        self._last_frame = None

    # --------------------------------------------------------------------- #
    # Platform generation helpers
    # --------------------------------------------------------------------- #
    def _generate_platform_until(self, target_x: float) -> None:
        """Procedurally extend the platform coverage up to a target x position."""
        while self._generated_until < target_x:
            segment_start = self._generated_until
            segment_end = segment_start + self.segment_length
            self._generated_until = segment_end

            # Keep the initial span free of gaps for stable starting behaviour.
            if segment_end <= self.initial_safe_span:
                # Still allow occasional orb placement in the safe span
                self._maybe_place_energy_orbs(segment_start, segment_end, None, None)
                continue

            if self.np_random.random() >= self.gap_probability:
                # Even if no gap, maybe place orbs for variety
                self._maybe_place_energy_orbs(segment_start, segment_end, None, None)
                continue

            available_length = (
                self.segment_length - (2 * self.segment_margin) - self.gap_width
            )
            if available_length <= 0:
                continue

            gap_start = (
                segment_start
                + self.segment_margin
                + self.np_random.random() * available_length
            )
            gap_end = gap_start + self.gap_width
            # Avoid overlaps by checking the previous gap.
            if self._gap_positions and gap_start <= self._gap_positions[-1][1]:
                continue
            self._gap_positions.append((gap_start, gap_end))
            # Place orbs around this segment/gap
            self._maybe_place_energy_orbs(
                segment_start, segment_end, gap_start, gap_end
            )

    def _prune_old_gaps(self, threshold: float) -> None:
        """Remove gaps that are far behind the current view."""
        while self._gap_positions and self._gap_positions[0][1] < threshold:
            self._gap_positions.pop(0)

    def _has_support(self) -> bool:
        """Return True if the character is standing on a platform (not a gap)."""
        x_left = self._character_x - (self.character_width / 2.0)
        x_right = self._character_x + (self.character_width / 2.0)
        for gap_start, gap_end in self._gap_positions:
            if x_right > gap_start and x_left < gap_end:
                return False
        return True

    # --------------------------------------------------------------------- #
    # Rendering helpers
    # --------------------------------------------------------------------- #
    def _render_observation(self) -> np.ndarray:
        """Return the current grayscale observation."""
        frame = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)

        camera_left = self._character_x - self.view_backward
        camera_width = self.view_backward + self.view_forward
        camera_bottom = self.ground_height - (self.view_height * 0.25)
        camera_top = camera_bottom + self.view_height

        pixels_per_unit_x = self.frame_width / camera_width
        pixels_per_unit_y = self.frame_height / (camera_top - camera_bottom)

        sky_level = int(self.frame_height * 0.6)
        frame[:sky_level, :] = 30
        frame[sky_level:, :] = 60

        # Draw the platform as a filled rectangle where ground exists.
        def world_to_pixel_x(x_value: float) -> float:
            return (x_value - camera_left) * pixels_per_unit_x

        def world_to_pixel_y(y_value: float) -> float:
            return (camera_top - y_value) * pixels_per_unit_y

        platform_top = int(
            np.clip(
                world_to_pixel_y(self.ground_height + self.platform_thickness),
                0,
                self.frame_height - 1,
            )
        )
        platform_bottom = int(
            np.clip(world_to_pixel_y(self.ground_height), 0, self.frame_height - 1)
        )
        if platform_top > platform_bottom:
            platform_top, platform_bottom = platform_bottom, platform_top

        for col in range(self.frame_width):
            world_x = camera_left + (col + 0.5) / pixels_per_unit_x
            if self._is_gap_at(world_x):
                continue
            frame[platform_top : platform_bottom + 1, col] = 180

        # Draw energy orbs
        for ox, oy, r in self._energy_orbs:
            px = int(
                np.clip((ox - camera_left) * pixels_per_unit_x, 0, self.frame_width - 1)
            )
            py = int(
                np.clip((camera_top - oy) * pixels_per_unit_y, 0, self.frame_height - 1)
            )
            pr = int(max(1, r * pixels_per_unit_x))
            self._draw_circle(frame, px, py, pr, 220)

        # Draw the character as a solid rectangle in the foreground.
        char_left = int(
            np.clip(
                world_to_pixel_x(self._character_x - (self.character_width / 2.0)),
                0,
                self.frame_width - 1,
            )
        )
        char_right = int(
            np.clip(
                world_to_pixel_x(self._character_x + (self.character_width / 2.0)),
                0,
                self.frame_width - 1,
            )
        )
        char_bottom = int(
            np.clip(world_to_pixel_y(self._character_y), 0, self.frame_height - 1)
        )
        char_top = int(
            np.clip(
                world_to_pixel_y(self._character_y + self.character_height),
                0,
                self.frame_height - 1,
            )
        )
        if char_top > char_bottom:
            char_top, char_bottom = char_bottom, char_top

        frame[char_top : char_bottom + 1, char_left : char_right + 1] = 255

        # HUD: timer top-left, energy top-right
        timer_text = f"{int(self._elapsed_time):d}s"
        energy_pct = int(round(self._energy * 100))
        energy_text = f"{energy_pct}%"
        self._draw_text(frame, 2, 2, timer_text, 255)
        energy_w = self._text_width(energy_text)
        self._draw_text(frame, self.frame_width - energy_w - 2, 2, energy_text, 255)

        self._last_frame = frame[:, :, np.newaxis]
        return self._last_frame

    # --------------------------------------------------------------------- #
    # Color rendering (for visualization only)
    # --------------------------------------------------------------------- #
    def _render_rgb_frame(self) -> np.ndarray:
        """Return a colorized RGB frame with a penguin sprite.

        This does not affect the observation returned to agents.
        """
        # Colors (R, G, B)
        SKY = (135, 206, 235)
        GROUND_BG = (36, 96, 36)
        GROUND = (60, 179, 113)
        ENERGY = (255, 215, 0)
        HUD = (255, 255, 255)

        h, w = self.frame_height, self.frame_width
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        camera_left = self._character_x - self.view_backward
        camera_width = self.view_backward + self.view_forward
        camera_bottom = self.ground_height - (self.view_height * 0.25)
        camera_top = camera_bottom + self.view_height

        px_per_x = self.frame_width / camera_width
        px_per_y = self.frame_height / (camera_top - camera_bottom)

        def world_to_px_x(x_value: float) -> float:
            return (x_value - camera_left) * px_per_x

        def world_to_px_y(y_value: float) -> float:
            return (camera_top - y_value) * px_per_y

        # Sky and ground background split similar to grayscale sky_level
        sky_level = int(self.frame_height * 0.6)
        rgb[:sky_level, :, :] = np.array(SKY, dtype=np.uint8)
        rgb[sky_level:, :, :] = np.array(GROUND_BG, dtype=np.uint8)

        # Platform band (only where no gap)
        platform_top = int(
            np.clip(
                world_to_px_y(self.ground_height + self.platform_thickness), 0, h - 1
            )
        )
        platform_bottom = int(np.clip(world_to_px_y(self.ground_height), 0, h - 1))
        if platform_top > platform_bottom:
            platform_top, platform_bottom = platform_bottom, platform_top
        if platform_bottom >= 0 and platform_top < h:
            ground_color = np.array(GROUND, dtype=np.uint8)
            for col in range(w):
                world_x = camera_left + (col + 0.5) / px_per_x
                if self._is_gap_at(world_x):
                    continue
                rgb[platform_top : platform_bottom + 1, col, :] = ground_color

        # Energy orbs
        for ox, oy, r in self._energy_orbs:
            px = int(np.clip((ox - camera_left) * px_per_x, 0, self.frame_width - 1))
            py = int(np.clip((camera_top - oy) * px_per_y, 0, self.frame_height - 1))
            pr = int(max(1, r * px_per_x))
            self._draw_circle_rgb(rgb, px, py, pr, ENERGY)

        # Character sprite overlay
        char_left = int(
            np.clip(
                world_to_px_x(self._character_x - (self.character_width / 2.0)),
                0,
                w - 1,
            )
        )
        char_right = int(
            np.clip(
                world_to_px_x(self._character_x + (self.character_width / 2.0)),
                0,
                w - 1,
            )
        )
        char_bottom = int(np.clip(world_to_px_y(self._character_y), 0, h - 1))
        char_top = int(
            np.clip(world_to_px_y(self._character_y + self.character_height), 0, h - 1)
        )
        if char_top > char_bottom:
            char_top, char_bottom = char_bottom, char_top

        # Ensure at least 1px tall/wide
        cw = max(1, char_right - char_left + 1)
        ch = max(1, char_bottom - char_top + 1)

        sprite_rgb, bg_rgb = self._get_sprite()
        if sprite_rgb is None:
            # Fallback: solid white rectangle (rare: Tk not available)
            rgb[char_top : char_top + ch, char_left : char_left + cw, :] = 255
        else:
            # Preserve aspect ratio: fit inside character box and center.
            sh, sw, _ = sprite_rgb.shape
            if sh <= 0 or sw <= 0:
                rgb[char_top : char_top + ch, char_left : char_left + cw, :] = 255
            else:
                scale = max(1e-6, min(cw / sw, ch / sh))
                tw = max(1, int(round(sw * scale)))
                th = max(1, int(round(sh * scale)))
                resized, mask = self._resize_sprite_with_mask(
                    sprite_rgb, bg_rgb, th, tw
                )
                # Center in the character rect and clip to bounds
                x0 = char_left + (cw - tw) // 2
                y0 = char_top + (ch - th) // 2
                x1 = x0 + tw
                y1 = y0 + th
                cx0 = max(char_left, x0)
                cy0 = max(char_top, y0)
                cx1 = min(char_left + cw, x1)
                cy1 = min(char_top + ch, y1)
                if cx1 > cx0 and cy1 > cy0:
                    sx0 = cx0 - x0
                    sy0 = cy0 - y0
                    sx1 = sx0 + (cx1 - cx0)
                    sy1 = sy0 + (cy1 - cy0)
                    patch = rgb[cy0:cy1, cx0:cx1, :]
                    sub_rgb = resized[sy0:sy1, sx0:sx1, :]
                    sub_mask = mask[sy0:sy1, sx0:sx1]
                    patch[sub_mask] = sub_rgb[sub_mask]
                    rgb[cy0:cy1, cx0:cx1, :] = patch

        # HUD text with background box and scaled font for readability
        timer_text = f"{int(self._elapsed_time):d}s"
        energy_pct = int(round(self._energy * 100))
        energy_text = f"{energy_pct}%"
        scale = 2
        self._draw_hud_label(rgb, 2, 2, timer_text, scale)
        energy_w = self._text_width(energy_text) * scale
        self._draw_hud_label(
            rgb, self.frame_width - energy_w - 4, 2, energy_text, scale
        )

        return rgb

    def _get_sprite(self) -> Tuple[np.ndarray | None, Tuple[int, int, int] | None]:
        """Load and cache the penguin sprite as an RGB numpy array.

        Uses tkinter.PhotoImage (std lib) to decode PNG without extra deps.
        Falls back to None if tkinter or image loading is unavailable.
        """
        if self._sprite_rgb is not None and self._sprite_bg_rgb is not None:
            return self._sprite_rgb, self._sprite_bg_rgb
        try:
            import tkinter as tk
        except Exception:
            return None, None

        try:
            # Resolve asset path from package resources
            with pkg_resources.as_file(
                pkg_resources.files("rlo.assets").joinpath("peng.png")
            ) as sprite_path:
                root = tk.Tk()
                root.withdraw()  # prevent window
                img = tk.PhotoImage(file=str(sprite_path))
                sw = img.width()
                sh = img.height()
                # Build numpy array of RGB from image.get
                arr = np.zeros((sh, sw, 3), dtype=np.uint8)
                # Use top-left pixel as background key (for transparency)
                bg = img.get(0, 0)
                if isinstance(bg, tuple):
                    bg_rgb = (int(bg[0]), int(bg[1]), int(bg[2]))
                else:
                    # Format like '#RRGGBB'
                    bg = str(bg)
                    bg_rgb = (
                        int(bg[1:3], 16),
                        int(bg[3:5], 16),
                        int(bg[5:7], 16),
                    )
                for y in range(sh):
                    for x in range(sw):
                        pix = img.get(x, y)
                        if isinstance(pix, tuple):
                            r, g, b = pix[:3]
                        else:
                            s = str(pix)
                            r, g, b = int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
                        arr[y, x, 0] = int(r)
                        arr[y, x, 1] = int(g)
                        arr[y, x, 2] = int(b)
                # Keep a reference to root to avoid early GC until cached
                self._sprite_rgb = arr
                self._sprite_bg_rgb = (int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]))
                try:
                    root.destroy()
                except Exception:
                    pass
                return self._sprite_rgb, self._sprite_bg_rgb
        except Exception:
            return None, None

    def _draw_circle_rgb(
        self, frame: np.ndarray, cx: int, cy: int, r: int, color: Tuple[int, int, int]
    ) -> None:
        h, w, _ = frame.shape
        r2 = r * r
        x0 = max(0, cx - r)
        x1 = min(w - 1, cx + r)
        y0 = max(0, cy - r)
        y1 = min(h - 1, cy + r)
        col = np.array(color, dtype=np.uint8)
        for py in range(y0, y1 + 1):
            dy = py - cy
            for px in range(x0, x1 + 1):
                dx = px - cx
                if dx * dx + dy * dy <= r2:
                    frame[py, px, :] = col

    # --- Sprite helpers ---
    def _resize_sprite_with_mask(
        self,
        sprite_rgb: np.ndarray,
        bg_rgb: Tuple[int, int, int],
        out_h: int,
        out_w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bilinear resize with a color-key alpha mask preserved via soft alpha.

        Returns (resized_rgb, alpha_mask_bool).
        """
        sh, sw, _ = sprite_rgb.shape
        # Alpha 1 for non-background
        alpha = ~(
            (sprite_rgb[:, :, 0] == bg_rgb[0])
            & (sprite_rgb[:, :, 1] == bg_rgb[1])
            & (sprite_rgb[:, :, 2] == bg_rgb[2])
        )
        alpha = alpha.astype(np.float32)

        # Source coordinates grid
        y_src = (np.linspace(0.0, sh - 1.0, out_h)).astype(np.float32)
        x_src = (np.linspace(0.0, sw - 1.0, out_w)).astype(np.float32)
        yy, xx = np.meshgrid(y_src, x_src, indexing="ij")

        y0 = np.floor(yy).astype(int)
        x0 = np.floor(xx).astype(int)
        y1 = np.clip(y0 + 1, 0, sh - 1)
        x1 = np.clip(x0 + 1, 0, sw - 1)
        wy = (yy - y0).astype(np.float32)
        wx = (xx - x0).astype(np.float32)

        def bilinear(channel: np.ndarray) -> np.ndarray:
            c00 = channel[y0, x0]
            c01 = channel[y0, x1]
            c10 = channel[y1, x0]
            c11 = channel[y1, x1]
            c0 = c00 * (1 - wx) + c01 * wx
            c1 = c10 * (1 - wx) + c11 * wx
            return c0 * (1 - wy) + c1 * wy

        r = bilinear(sprite_rgb[:, :, 0].astype(np.float32))
        g = bilinear(sprite_rgb[:, :, 1].astype(np.float32))
        b = bilinear(sprite_rgb[:, :, 2].astype(np.float32))
        a = bilinear(alpha)
        resized = np.stack([r, g, b], axis=-1)
        resized = np.clip(resized, 0, 255).astype(np.uint8)
        mask = a > 0.5
        return resized, mask

    # --- HUD helpers ---
    def _draw_rect_alpha(
        self,
        frame: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: Tuple[int, int, int],
        alpha: float,
    ) -> None:
        h, w, _ = frame.shape
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if x1 <= x0 or y1 <= y0:
            return
        col = np.array(color, dtype=np.float32)
        patch = frame[y0:y1, x0:x1, :].astype(np.float32)
        patch = (1 - alpha) * patch + alpha * col
        frame[y0:y1, x0:x1, :] = np.clip(patch, 0, 255).astype(np.uint8)

    def _draw_hud_label(
        self, frame: np.ndarray, x: int, y: int, text: str, scale: int
    ) -> None:
        pad = 2
        w = self._text_width(text) * scale
        h = 5 * scale
        # Background
        self._draw_rect_alpha(
            frame, x - pad, y - pad, x + w + pad, y + h + pad, (0, 0, 0), 0.45
        )
        # Shadow
        self._draw_text_rgb(frame, x + 1, y + 1, text, (0, 0, 0), scale)
        # Text
        self._draw_text_rgb(frame, x, y, text, (255, 255, 255), scale)

    def _draw_text_rgb(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        text: str,
        color: Tuple[int, int, int],
        scale: int = 1,
    ) -> None:
        h, w, _ = frame.shape
        col = np.array(color, dtype=np.uint8)
        for i, ch in enumerate(text):
            glyph = self._FONT.get(ch)
            if glyph is None:
                continue
            gx = x + i * (3 * scale + 1)
            for r, bits in enumerate(glyph):
                for rr in range(scale):
                    py = y + r * scale + rr
                    if py < 0 or py >= h:
                        continue
                    for c in range(3):
                        if (bits >> (2 - c)) & 1:
                            for cc in range(scale):
                                px = gx + c * scale + cc
                                if 0 <= px < w:
                                    frame[py, px, :] = col

    def _is_gap_at(self, x_position: float) -> bool:
        """Return True if the given world x position lies within a gap."""
        for gap_start, gap_end in self._gap_positions:
            if gap_start <= x_position <= gap_end:
                return True
        return False

    # --------------------------------------------------------------------- #
    # Energy/orb management and HUD helpers
    # --------------------------------------------------------------------- #
    def _get_metrics_vector(self) -> np.ndarray:
        return np.array(
            [self._energy, self.energy_max_overfill, float(self._elapsed_time)],
            dtype=np.float32,
        )

    def _handle_energy_orbs_collision(self) -> Tuple[float, float, float] | None:
        # Character AABB
        x_left = self._character_x - (self.character_width / 2.0)
        x_right = self._character_x + (self.character_width / 2.0)
        y_bottom = self._character_y
        y_top = self._character_y + self.character_height

        next_orbs: List[Tuple[float, float, float]] = []
        for ox, oy, r in self._energy_orbs:
            cx = min(max(ox, x_left), x_right)
            cy = min(max(oy, y_bottom), y_top)
            dx = ox - cx
            dy = oy - cy
            if dx * dx + dy * dy <= r * r:
                # return the orb that was collected
                return (ox, oy, r)
            else:
                next_orbs.append((ox, oy, r))
        self._energy_orbs = next_orbs
        return None

    def _maybe_place_energy_orbs(
        self,
        segment_start: float,
        segment_end: float,
        gap_start: float | None,
        gap_end: float | None,
    ) -> None:
        # Randomly decide none/sparse/cluster
        roll = self.np_random.random()
        if roll < 0.4:
            return
        cluster = roll > 0.8
        count = int(
            self.np_random.integers(2, 5) if cluster else self.np_random.integers(1, 3)
        )
        radius = 0.18

        for _ in range(count):
            if (
                gap_start is not None
                and gap_end is not None
                and self.np_random.random() < 0.5
            ):
                # Place relative to gap: before/after or above
                if self.np_random.random() < 0.5:
                    x0 = segment_start + 0.5
                    x1 = max(segment_start + 0.6, gap_start - 0.3)
                else:
                    x0 = min(gap_end + 0.3, segment_end - 0.6)
                    x1 = segment_end - 0.5
            else:
                x0 = segment_start + 0.5
                x1 = segment_end - 0.5
            if x1 <= x0:
                x = segment_start + self.np_random.random() * (
                    segment_end - segment_start
                )
            else:
                x = x0 + self.np_random.random() * (x1 - x0)

            # Vertical placement variety
            mode = self.np_random.random()
            if mode < 0.35:
                y = self.ground_height + self.platform_thickness + radius + 0.02
            elif mode < 0.7:
                y = self.ground_height + 0.6 * self.character_height
            elif mode < 0.9:
                y = self.ground_height + 1.2 * self.character_height
            else:
                y = self.ground_height + 1.8 * self.character_height
            self._energy_orbs.append((x, y, radius))

    # Minimal pixel font and rasterizers
    _FONT = {
        "0": [0b111, 0b101, 0b101, 0b101, 0b111],
        "1": [0b010, 0b110, 0b010, 0b010, 0b111],
        "2": [0b111, 0b001, 0b111, 0b100, 0b111],
        "3": [0b111, 0b001, 0b111, 0b001, 0b111],
        "4": [0b101, 0b101, 0b111, 0b001, 0b001],
        "5": [0b111, 0b100, 0b111, 0b001, 0b111],
        "6": [0b111, 0b100, 0b111, 0b101, 0b111],
        "7": [0b111, 0b001, 0b010, 0b010, 0b010],
        "8": [0b111, 0b101, 0b111, 0b101, 0b111],
        "9": [0b111, 0b101, 0b111, 0b001, 0b111],
        "%": [0b110, 0b110, 0b001, 0b011, 0b011],
        "s": [0b000, 0b111, 0b100, 0b111, 0b001],
    }

    def _text_width(self, text: str) -> int:
        return max(0, len(text) * 4 - 1)

    def _draw_text(
        self, frame: np.ndarray, x: int, y: int, text: str, color: int
    ) -> None:
        h, w = frame.shape
        for i, ch in enumerate(text):
            glyph = self._FONT.get(ch)
            if glyph is None:
                continue
            gx = x + i * 4
            for r, bits in enumerate(glyph):
                py = y + r
                if py < 0 or py >= h:
                    continue
                for c in range(3):
                    if (bits >> (2 - c)) & 1:
                        px = gx + c
                        if 0 <= px < w:
                            frame[py, px] = color

    def _draw_circle(
        self, frame: np.ndarray, cx: int, cy: int, r: int, color: int
    ) -> None:
        h, w = frame.shape
        r2 = r * r
        x0 = max(0, cx - r)
        x1 = min(w - 1, cx + r)
        y0 = max(0, cy - r)
        y1 = min(h - 1, cy + r)
        for py in range(y0, y1 + 1):
            dy = py - cy
            for px in range(x0, x1 + 1):
                dx = px - cx
                if dx * dx + dy * dy <= r2:
                    frame[py, px] = color
