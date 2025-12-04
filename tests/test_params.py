import os
import yaml
import pytest
from rlo.params import ParamReader

class TestClass:
    pass

class AnotherClass:
    pass

def test_param_reader(tmp_path):
    # Setup
    config_path = tmp_path / "params.yaml"
    config_data = {
        "TestClass": {
            "param1": 10,
            "param2": "hello"
        },
        "AnotherClass": {
            "param3": 3.14
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)

    reader = ParamReader.get_instance()
    reader.clear()
    reader.load(str(config_path))

    # Test 1: Basic retrieval
    obj = TestClass()
    assert reader.get(obj, "param1") == 10
    assert reader.get(obj, "param2") == "hello"
    assert reader.get(AnotherClass, "param3") == 3.14

    # Test 2: Default value
    assert reader.get(obj, "missing_param", default=999) == 999

    # Test 3: Overrides
    reader.set_overrides(param1=20, new_param=100)
    assert reader.get(obj, "param1") == 20  # Override wins
    assert reader.get(obj, "new_param") == 100 # Global override
    
    reader.clear()
