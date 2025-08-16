"""Base configuration classes and utilities for DIET finetuning."""

import dataclasses
import torch
from typing import Dict, Any, Optional, Tuple, Union, List
from abc import ABC, abstractmethod


# Hardware settings - single source of truth
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global defaults - centralized
GLOBAL_DEFAULTS = {
    "temperature": 3.0,
    "enable_wandb": True,
    "wandb_project": "DIET-Finetuning",
}

# Valid options - single source of truth
VALID_TRAINING_MODES = ["combined", "diet_only", "probe_only"]


@dataclasses.dataclass
class BaseConfig(ABC):
    """Base configuration class with common functionality."""

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary, filtering invalid fields."""
        valid_fields = {field.name for field in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return dataclasses.asdict(self)

    def update(self, **kwargs) -> "BaseConfig":
        """Create new instance with updated values."""
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return self.from_dict(current_dict)

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration values."""
        pass


def validate_training_mode(mode: str) -> None:
    """Validate training mode."""
    if mode not in VALID_TRAINING_MODES:
        raise ValueError(
            f"Invalid training mode: {mode}. Must be one of {VALID_TRAINING_MODES}"
        )
