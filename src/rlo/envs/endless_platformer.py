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
        gap_probability: float = 0.7,
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
        self.character_width = 0.2
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

        # Apply forward boost when jumping
        if action == self.JUMP:
            forward_speed += self.forward_boost

        # Horizontal movement only when forward action is taken
        previous_x = self._character_x
        if action == self.FORWARD or action == self.JUMP:
            self._character_x += forward_speed * self.dt

        # Jump initiation.
        if action == self.JUMP and self._on_ground:
            self._character_vy = self.jump_velocity
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
        reward = self._character_x - previous_x
        terminated = self._energy <= 0.0
        truncated = False
        info = {
            "x_position": self._character_x,
            "y_position": self._character_y,
            "x_velocity": forward_speed,
            "y_velocity": self._character_vy,
            "on_ground": self._on_ground,
            "fell": fell,
            "energy": self._energy,
            "energy_max": self.energy_max_overfill,
            "elapsed_time_s": self._elapsed_time,
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Return an RGB frame when render_mode is 'rgb_array'."""
        if self.render_mode != "rgb_array":
            raise ValueError("Only the 'rgb_array' render mode is supported.")
        if self._last_frame is None:
            _ = self._render_observation()
        gray = self._last_frame
        return np.repeat(gray, 3, axis=2)

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
