import numpy as np
import torch
import pytest

from rlo.policies import ParamNonLinearPolicy_ArgMax


def test_param_count_matches_model():
    pol = ParamNonLinearPolicy_ArgMax(n_actions=3, n_features=7)

    # count params from pytorch
    torch_param_total = sum(p.numel() for p in pol._model.parameters())
    class_param_total = pol.get_params().size

    assert torch_param_total == class_param_total


def test_get_set_roundtrip():
    pol = ParamNonLinearPolicy_ArgMax(n_actions=3, n_features=5)

    # original params
    p1 = pol.get_params().copy()

    # load back
    pol.set_params(p1)
    p2 = pol.get_params().copy()

    assert np.allclose(p1, p2)


def test_set_params_changes_output():
    pol = ParamNonLinearPolicy_ArgMax(n_actions=3, n_features=4)

    x = np.array([0.5, -1.0, 2.0, 0.1], dtype=np.float32)

    # original output
    a1, info1 = pol.act(x, {})

    # tweak params slightly
    params = pol.get_params()
    params = params + 0.01
    pol.set_params(params)

    a2, info2 = pol.act(x, {})

    # logits should change
    assert not np.allclose(info1["logits"],
                           info2["logits"])


def test_set_params_loads_known_values():
    pol = ParamNonLinearPolicy_ArgMax(n_actions=2, n_features=3)

    flat = np.arange(pol.get_params().size, dtype=np.float32)

    pol.set_params(flat)
    p2 = pol.get_params()

    assert np.allclose(flat, p2)


def test_layer1_contributions_shape():
    pol = ParamNonLinearPolicy_ArgMax(n_actions=4, n_features=6)

    x = np.random.randn(6).astype(np.float32)
    action, info = pol.act(x, {})

    contrib = info["contributions"]

    # should be (hidden_dim, n_features)
    expected_shape = pol._model.fc1.weight.shape

    assert contrib.shape == expected_shape
