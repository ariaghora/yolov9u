import os
from argparse import ArgumentParser
from glob import glob
from typing import Callable, List, Literal, Optional, Set

import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image
from torch.utils.data.dataset import Dataset

from yolov9u.config import ModelConfig, TrainingConfig
from yolov9u.models import YOLODetector


class DetectionDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: Optional[str],
        points_format: Literal["xywh", "xyxy", "points"],
        transform_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """
        For now, we assume that the labels are in jpg, and the txt is
        in YOLO xywh format.
        """
        self.points_format = points_format
        self.label_dir = label_dir
        self.transform_fn = transform_fn

        logger.info("scanning images and labels...")
        # take basenames without extension. We assume the basename is identical for
        # the image and its corresponding label.
        image_dirs = glob(os.path.join(image_dir, "*.jpg"))
        assert len(image_dirs) > 0, "No image found"
        image_basenames = set([s.split("/")[-1].rstrip(".jpg") for s in image_dirs])
        logger.info(f"found {len(image_basenames)} images")

        label_basenames: Optional[Set[str]] = None
        label_dirs: Optional[List[str]] = None
        if self.label_dir is not None:
            label_dirs = glob(os.path.join(self.label_dir, "*.txt"))
            label_basenames = set([s.split("/")[-1].rstrip(".txt") for s in label_dirs])
            assert len(label_dirs) > 0, "No label found"
            logger.info(f"found {len(label_basenames)} labels")

        # Get the number of image having no label, or label having no image, i.e.,
        # we take symmetric difference between image_basenames and label_basenames
        difference_count = 0
        if label_basenames:
            difference_count = len(image_basenames ^ label_basenames)

        if difference_count > 0:
            logger.warning(
                f"found {difference_count} images/labels with no associated labels/images"
            )
            if label_basenames:
                common_basenames = image_basenames & label_basenames
            else:
                common_basenames = image_basenames
            image_basenames = common_basenames
            label_basenames = common_basenames
            logger.warning(f"only {len(common_basenames)} samples will be used")

        # Collect the final set of image and label filenames
        image_common_path = os.path.commonpath(image_dirs)
        self.image_dirs = [
            os.path.join(image_common_path, fn + ".jpg")
            for fn in sorted(image_basenames)
        ]

        if label_dirs and label_basenames:
            label_common_path = os.path.commonpath(label_dirs)
            self.label_dirs = [
                os.path.join(label_common_path, fn + ".txt")
                for fn in sorted(label_basenames)
            ]
        else:
            self.label_dirs = None

    def __getitem__(self, idx: int):
        if self.label_dir:
            return self._get_image(idx), self._get_label(idx)
        return self._get_image(idx)

    @logger.catch
    def _get_image(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_dirs[idx])
        if self.transform_fn:
            image_tensor = self.transform_fn(image)
        else:
            image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1)
        return image_tensor

    @logger.catch
    def _get_label(self, idx: int) -> torch.Tensor:
        # We manually parse annotations line by line, to ensure that we are
        # dealing with yolo dataset rather than polygon points
        assert self.label_dirs is not None

        label_filename = self.label_dirs[idx]
        with open(label_filename) as f:
            annotation_lines = f.read().strip().split("\n")

        parsed_annotation_list = []
        for i, line in enumerate(annotation_lines):
            parts = [float(v) for v in line.strip().split(" ")]
            class_label = parts[0]
            values = parts[1:]
            if self.points_format == "points":
                assert (
                    len(values) % 2 == 0
                ), f"points at row {i + 1} in {label_filename} cannot be reshaped into (-1, 2) to represents coordinates"

                points = np.array(values).reshape(-1, 2)
                x0, y0 = points.min(0)
                x1, y1 = points.max(0)
                w, h = x1 - x0, y1 - y0
                x, y = x0 + w / 2, y0 + h / 2
            elif self.points_format == "xywh":
                assert (
                    len(values) == 4
                ), f"xywh format requires 4 values, found {len(values)} at row {i + 1} in {label_filename}"
                x, y, w, h = values
            else:
                raise ValueError(f"unrecognized format: {self.points_format}")
            parsed_annotation_list.append([class_label, x, y, w, h])

        annotation = np.array(parsed_annotation_list)
        return torch.FloatTensor(annotation)


def train():
    pass


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

    dataset_training = DetectionDataset(
        image_dir=training_config.training_image_dir,
        label_dir=training_config.training_label_dir,
        points_format="points",
    )

    model = YOLODetector(model_config).float()
    if training_config.with_pretrained_weight:
        logger.info(
            f"prertrained model is used: {training_config.pretrained_weight_path}"
        )
        model.load_state_dict(
            torch.load(training_config.pretrained_weight_path, weights_only=True)
        )
