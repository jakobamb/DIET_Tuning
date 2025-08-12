"""Training configuration for DIET finetuning."""

import dataclasses
import torch
from typing import Dict, Any, Optional, Tuple, Union


# Hardware settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default training settings
DEFAULT_TRAINING_CONFIG = {
    "num_epochs": 30,
    "learning_rate": 5e-4,
    "weight_decay": 0.05,
    "label_smoothing": 0.3,
    "training_mode": "combined",  # Options: "combined", "diet_only", "probe_only"
    "eval_frequency": 5,
}

# Default paths
DEFAULT_PATHS = {
    "checkpoints": "checkpoints", 
    "results": "results",
    "data": "data"
}

# Default logging settings
DEFAULT_LOGGING_CONFIG = {
    "project": "DIET-Finetuning", 
    "enable_wandb": True
}


@dataclasses.dataclass
class TrainerConfig:
    """Configuration for the DIETTrainer class."""
    # Model configuration
    backbone_type: str
    model_size: str
    num_classes: int
    projection_dim: int = 256
    temperature: float = 3.0
    embedding_dim: Optional[int] = None

    # Data configuration
    dataset_name: str = "cifar10"
    batch_size: int = 32
    limit_data: int = 1000
    is_rgb: bool = True
    input_size: int = 32
    dataset_mean: Union[Tuple[float, ...], Tuple[float]] = (0.5,)
    dataset_std: Union[Tuple[float, ...], Tuple[float]] = (0.5,)
    num_diet_classes: int = 100
    
    # Training configuration
    num_epochs: int = 30
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.3
    training_mode: str = "combined"
    eval_frequency: int = 5
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    
    # Logging settings
    enable_wandb: bool = True
    wandb_project: str = "DIET-Finetuning"

    def __post_init__(self):
        """Validate and set derived attributes after initialization."""
        # Validate training mode
        valid_modes = ["combined", "diet_only", "probe_only"]
        if self.training_mode not in valid_modes:
            raise ValueError(
                f"Invalid training mode: {self.training_mode}. "
                f"Must be one of {valid_modes}"
            )
            
        # Import at runtime to avoid circular imports
        from config.model_config import get_model_embedding_dim
        from config.data_config import get_dataset_stats
        
        # Set embedding dimension if not provided
        if self.embedding_dim is None:
            self.embedding_dim = get_model_embedding_dim(
                self.backbone_type, self.model_size
            )

        # Set dataset stats if not already set
        if self.dataset_mean == (0.5,) and self.dataset_std == (0.5,):
            dataset_info = get_dataset_stats(self.dataset_name)
            self.input_size = dataset_info["input_size"]
            self.is_rgb = dataset_info["is_rgb"]
            self.dataset_mean = dataset_info["mean"]
            self.dataset_std = dataset_info["std"]
            # Only update num_classes if not explicitly set or invalid
            if self.num_classes <= 0:
                self.num_classes = dataset_info["num_classes"]

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainerConfig":
        """Create a TrainerConfig from a dictionary of parameters.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            TrainerConfig instance
        """
        valid_fields = {field.name for field in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary.
        
        Returns:
            Dictionary representation of this config
        """
        return dataclasses.asdict(self)
