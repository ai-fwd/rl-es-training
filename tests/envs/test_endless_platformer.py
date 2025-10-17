import numpy as np

from rlo.envs.endless_platformer import EndlessPlatformerEnv


def test_endless_platformer_shapes_and_dynamics():
    env = EndlessPlatformerEnv(render_mode="rgb_array")
    observation, info = env.reset(seed=123)

    assert observation["image"].shape == env.observation_space["image"].shape
    assert info["on_ground"] is True

    for action in range(env.action_space.n):
        obs, reward, terminated, truncated, step_info = env.step(action)
        assert obs["image"].shape == env.observation_space["image"].shape
        assert np.isfinite(reward)
        assert terminated is False and truncated is False
        assert "x_position" in step_info
        assert "energy" in step_info and "elapsed_time_s" in step_info

    frame = env.render()
    assert frame.shape[:2] == env.observation_space["image"].shape[:2]
    assert frame.shape[2] == 3

    env.close()
