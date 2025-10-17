"""Endless platform walker environment compatible with Gymnasium.

The environment models a side-scrolling character that advances over a
platform with fixed-width gaps that appear at random intervals. The agent
receives grayscale pixel observations of the current viewport and can choose
between three discrete actions:

    0 - NOOP: no movement
    1 - FORWARD: move forward
    2 - JUMP: initiate a jump when on the platform (also grants forward burst)

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

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        gap_width: float = 2.2,
        gap_probability: float = 0.7,
        segment_length: float = 8.0,
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
        self.character_width = 0.8
        self.character_height = 1.4
        self.ground_height = 0.0
        self.platform_thickness = 0.5
        self.fall_threshold = -5.0

        # Procedural generation parameters.
        self.gap_width = gap_width
        self.gap_probability = gap_probability
        self.segment_length = segment_length
        self.segment_margin = 1.5
        self.initial_safe_span = 12.0
        self.cleanup_margin = 5.0

        # Viewport configuration for rendering.
        self.frame_width = 160
        self.frame_height = 90
        self.view_backward = 3.5
        self.view_forward = 6.5
        self.view_height = 8.0

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.frame_height, self.frame_width, 1),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(3)

        self._character_x = 0.0
        self._character_y = 0.0
        self._character_vy = 0.0
        self._on_ground = True
        self._last_safe_x = 0.0

        self._gap_positions: List[Tuple[float, float]] = []
        self._generated_until = 0.0

        self._last_frame: np.ndarray | None = None

    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict | None = None,
    ) -> Tuple[np.ndarray, Dict]:
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
        self._generate_platform_until(self._character_x + self.view_forward + 10.0)

        self._last_frame = None
        observation = self._render_observation()
        info = {
            "x_position": self._character_x,
            "y_position": self._character_y,
            "on_ground": self._on_ground,
            "gaps": tuple(self._gap_positions),
        }
        return observation, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
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

        # Apply gravity.
        self._character_vy += self.gravity * self.dt
        self._character_y += self._character_vy * self.dt

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

        observation = self._render_observation()
        reward = self._character_x - previous_x
        terminated = False
        truncated = False
        info = {
            "x_position": self._character_x,
            "y_position": self._character_y,
            "x_velocity": forward_speed,
            "y_velocity": self._character_vy,
            "on_ground": self._on_ground,
            "fell": fell,
        }
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Render the current frame."""
        if self.render_mode != "rgb_array":
            raise ValueError("Only the 'rgb_array' render mode is supported.")
        if self._last_frame is None:
            self._last_frame = self._render_observation()
        # Repeat grayscale channel to RGB.
        return np.repeat(self._last_frame, 3, axis=-1)

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
                continue

            if self.np_random.random() >= self.gap_probability:
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

        self._last_frame = frame[:, :, np.newaxis]
        return self._last_frame

    def _is_gap_at(self, x_position: float) -> bool:
        """Return True if the given world x position lies within a gap."""
        for gap_start, gap_end in self._gap_positions:
            if gap_start <= x_position <= gap_end:
                return True
        return False
