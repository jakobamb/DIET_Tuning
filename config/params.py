"""Default parameters for DIET finetuning."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DietParams:
    """Default parameters for DIET finetuning."""

    # Model parameters
    backbone_type: str = "simclr"
    model_size: str = "resnet50-1x"

    # Dataset parameters
    dataset_name: str = "cifar10"
    limit_data: int = 1000

    # Training parameters
    num_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    da_strength: int = 1
    resume_from: Optional[str] = None
    training_mode: str = "combined"

    # DIET parameters
    label_smoothing: float = 0.3
    num_diet_classes: int = 100
    projection_dim: int = 256

    # Evaluation/utility parameters
    run_sanity_check: bool = True
    expected_threshold: float = 0.85


# Create a single instance of the parameters
DEFAULT_PARAMS = DietParams()
