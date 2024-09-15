from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ModelConfig:
    backbone: List[Any]
    head: List[Any]
    class_count: int
    depth_multiple: float = 1.0
    width_multiple: float = 1.0
    input_channels: int = 3
    anchors: int = 3
    activation: Optional[str] = None
    inplace: bool = True


@dataclass
class TrainingConfig:
    with_pretrained_weight: bool
    pretrained_weight_path: str
    minibatch_size: int

    training_image_dir: str
    training_label_dir: str
    validation_image_dir: str
    validation_label_dir: str

    label_extension: str
    label_value_separator: str
