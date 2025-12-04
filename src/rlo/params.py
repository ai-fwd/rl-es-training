import yaml
import os
from typing import Any, Dict, Optional, Type, Union

class ParamReader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ParamReader, cls).__new__(cls)
            cls._instance._config = {}
            cls._instance._overrides = {}
        return cls._instance

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls()
        return cls._instance

    def load(self, yaml_path: str) -> None:
        """Load configuration from a YAML file."""
        if not os.path.exists(yaml_path):
            print(f"Warning: Config file {yaml_path} not found. Using empty config.")
            return

        with open(yaml_path, 'r') as f:
            try:
                data = yaml.safe_load(f)
                if data:
                    self._config.update(data)
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file {yaml_path}: {e}")

    def set_overrides(self, **kwargs) -> None:
        """Set global overrides for parameters."""
        self._overrides.update(kwargs)

    def get(self, obj_or_class: Union[object, Type], param_name: str, default: Any = None) -> Any:
        """
        Get a parameter value.
        Priority:
        1. Global Overrides (CLI args)
        2. YAML Config (Class specific)
        3. Default value
        """
        # 1. Check overrides
        if param_name in self._overrides:
            return self._overrides[param_name]

        # 2. Check YAML config
        # Get class name
        if isinstance(obj_or_class, type):
            class_name = obj_or_class.__name__
        else:
            class_name = obj_or_class.__class__.__name__

        if class_name in self._config:
            if param_name in self._config[class_name]:
                return self._config[class_name][param_name]

        # 3. Return default
        if default is not None:
            return default
            
        raise ValueError(f"Parameter '{param_name}' not found for {class_name} and no default provided.")

    def dump(self, yaml_path: str) -> None:
        """Save the current effective configuration (including overrides) to a YAML file."""
        # We want to save a merged view: config + overrides
        # However, overrides are flat, while config is hierarchical.
        # For simplicity, we will just dump the loaded config for now, 
        # OR we could try to merge overrides back into the structure if we knew where they belong.
        # Given the requirement "save a copy of the yaml parameters", dumping the _config is safest.
        # If we want to persist CLI overrides, we'd need to know which class they belong to.
        # For now, let's dump the _config. 
        # TODO: If overrides are critical to persist, we might need a mapping or heuristic.
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def clear(self):
        """Reset config and overrides (mostly for testing)."""
        self._config = {}
        self._overrides = {}
