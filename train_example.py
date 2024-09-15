from argparse import ArgumentParser

import torch
import yaml

from yolov9u.config import ModelConfig, TrainingConfig
from yolov9u.models import YOLODetector
from loguru import logger


def train(): ...


if __name__ == "__main__":
    parser = ArgumentParser(prog="YOLOv9 training program")
    parser.add_argument("-m", "--model-config", required=True)
    parser.add_argument("-t", "--training-config", required=True)
    parser.add_argument("-e", "--experiment-name", required=True)
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_config = ModelConfig(**yaml.load(f, yaml.SafeLoader))

    with open(args.training_config) as f:
        training_config = TrainingConfig(**yaml.load(f, yaml.SafeLoader))

    model = YOLODetector(model_config).float()
    if training_config.with_pretrained_weight:
        logger.info(f"prertrained model is used: {training_config.pretrained_weight_path}")
        model.load_state_dict(
            torch.load(training_config.pretrained_weight_path, weights_only=True)
        )
