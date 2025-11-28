import tkinter as tk
import numpy as np
import pytest
from unittest.mock import MagicMock
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))

from play_endless_platformer import ScratchpadPanel

def test_scratchpad_update_jepa():
    # Mock Tkinter root
    root = tk.Tk()
    
    # Mock PolicyBundle and Action Labels
    bundle = MagicMock()
    action_labels = {0: "noop", 1: "right", 2: "jump", 3: "eat"}
    
    panel = ScratchpadPanel(root, bundle, action_labels)
    
    # Mock data from JEPA policy
    step = 10
    action_id = 1
    metrics = np.array([50.0]) # energy
    info_before = {"can_eat": True}
    info_after = {"on_ground": True}
    
    diagnostics = {
        "selected_action": 1,
        "logits": np.array([0.1, 0.5, 0.2, 0.2]),
        "curiosity_scores": np.array([0.01, 0.05, 0.02, 0.02]),
        "validity": 0.95,
        "features": np.zeros(8) # dummy features
    }
    
    # Run update
    try:
        panel.update(step, action_id, diagnostics, metrics, info_before, info_after)
    except Exception as e:
        pytest.fail(f"Scratchpad update failed with JEPA data: {e}")
    
    # Verify UI variables
    assert "Validity: 0.950" in panel._validity_var.get()
    assert "RIGHT" in panel._action_var.get()
    
    # Verify curiosity scores in features label
    features_text = panel._features_label.cget("text")
    assert "Curiosity Scores:" in features_text
    assert "RIGHT: +0.050" in features_text
    
    root.destroy()

if __name__ == "__main__":
    test_scratchpad_update_jepa()
    print("Scratchpad JEPA test passed!")
