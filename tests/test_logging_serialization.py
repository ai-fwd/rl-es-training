import numpy as np

from rlo.utils.logging import GenerationStats


def test_generation_stats_serializes_curiosity_fields():
    policy_info = [
        {
            "selected_action": 1,
            "logits": np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32),
            "curiosity_scores": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
            "validity": 0.9,
        }
    ]

    stats = GenerationStats(
        generation=0,
        wall_time_s=1.0,
        best_reward=2.5,
        sigma=0.1,
        policy_info=policy_info,
    )

    payload = stats.to_dict()
    serialized_action = payload["best_policy"]["trace"][0]

    assert serialized_action["curiosity_scores"] == [0.5, 0.6, 0.7, 0.8]
    assert serialized_action["validity"] == 0.9
