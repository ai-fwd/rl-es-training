"""Agent utilities and policy definitions for RLO environments."""
from .param_base import (
    Policy,
)
from .param_linear import (
    ParamLinearPolicy,
)

# A simple registry so serialized bundles can record and later lookup
# which concrete policy class to instantiate when loading.
POLICY_REGISTRY: dict[str, type] = {}

def register_policy(cls: type) -> type:
    """Register a policy class in the package-level registry.

    Stores by fully-qualified class name (module.ClassName).
    Use as a decorator on concrete policy classes.
    """
    key = f"{cls.__module__}.{cls.__name__}"
    POLICY_REGISTRY[key] = cls
    return cls


def get_policy_class(fqcn: str) -> type | None:
    """Return a registered policy class by fully-qualified name, or None."""
    return POLICY_REGISTRY.get(fqcn)


from .param_nonlinear_base import ParamNonLinearPolicy

from .param_nonlinear import (
    ParamNonLinearPolicy_ArgMax,
    ParamNonLinearPolicy_Stochastic,
)

__all__ = [
    "Policy",
    "ParamLinearPolicy",
    "ParamNonLinearPolicy",
    "ParamNonLinearPolicy_ArgMax",
    "ParamNonLinearPolicy_Stochastic",
    "register_policy",
    "get_policy_class",
    "POLICY_REGISTRY",
]
