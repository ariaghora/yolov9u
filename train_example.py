import os
import random
from argparse import ArgumentParser
from glob import glob
from typing import Any, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from loguru import logger
from PIL import Image
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as F
from tqdm import tqdm

from yolov9u.config import ModelConfig, TrainingConfig
from yolov9u.models import YOLODetector
from yolov9u.loss import YOLOLoss


def yolo_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for YOLO DataLoader.

    Regular collcate_fn cannot handle YOLO target, since an image
    may have different number of bounding boxes. As for the lost of image
    itself, we can simply stack like what the default collate_fn do.

    I followed the original code to put image index within a batch. Extra
    first column was created in the DetectionDataset class' __getitem__.
    Not doing this will cause shape mismatch error.
    """
    images, targets = zip(*batch)
    for i, target in enumerate(targets):
        target[:, 0] = i

    # Stack images into a single tensor
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)

    return images, targets


class YOLOTransform:
    def __init__(self, size: int):
        self.size = size

    def letterbox(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = image.shape[1:]
        scale = min(self.size / h, self.size / w)
        nh, nw = int(h * scale), int(w * scale)

        # Resize image
        image = F.resize(image, [nh, nw])

        # Create new image with gray padding
        new_image = torch.full((3, self.size, self.size), 0.5)
        dx, dy = (self.size - nw) // 2, (self.size - nh) // 2
        new_image[:, dy : dy + nh, dx : dx + nw] = image

        # Adjust bounding boxes
        if target.numel() > 0:
            # Convert YOLO format to pixel coordinates
            target[:, [1, 3]] *= w
            target[:, [2, 4]] *= h

            # Apply scale and offset
            target[:, 1] = target[:, 1] * scale + dx
            target[:, 2] = target[:, 2] * scale + dy
            target[:, 3] *= scale
            target[:, 4] *= scale

            # Convert back to normalized coordinates
            target[:, [1, 3]] /= self.size
            target[:, [2, 4]] /= self.size

            target_out = torch.zeros(len(target), 6)
            target_out[:, 1:] = target
            target = target_out

        return new_image, target

    def __call__(
        self, image: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Letterbox
        image, target = self.letterbox(image, target)

        # Random horizontal flip
        if random.random() > 0.5:
            image = F.hflip(image)
            if target.numel() > 0:
                target[:, 1] = 1 - target[:, 1]  # flip x_center

        # Color jitter
        image = transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
        )(image)

        # Normalize
        image = image / 255.0

        return image, target


class DetectionDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: Optional[str],
        points_format: Literal["xywh", "xyxy", "points"],
        transform_fn: Optional[Any] = None,
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

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.label_dir:
            image, label = self._get_image(idx), self._get_label(idx)
            if self.transform_fn:
                image_tensor, label = self.transform_fn(image, label)
            else:
                image_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1)
            return image_tensor, label
        return self._get_image(idx)

    def __len__(self):
        return len(self.image_dirs)

    @logger.catch
    def _get_image(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_dirs[idx])
        return F.to_tensor(image)

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
        transform_fn=YOLOTransform(640),
    )

    dataloader_training = DataLoader(
        dataset_training,
        shuffle=True,
        batch_size=training_config.minibatch_size,
        drop_last=True,
        collate_fn=yolo_collate_fn,
    )

    model = YOLODetector(model_config)
    if training_config.with_pretrained_weight:
        logger.info(
            f"prertrained model is used: {training_config.pretrained_weight_path}"
        )
        model.load_state_dict(
            torch.load(training_config.pretrained_weight_path, weights_only=True)
        )
    model = model.to(training_config.device).train()
    if training_config.device == "cuda":
        model = model.half()  # I'm GPU-poor

    model = nn.DataParallel(model)

    loss_func = YOLOLoss(
        model_config.class_count, model.module.model[-1].stride, training_config.device
    )
    loss_scaler = None
    if training_config.device == "cuda":
        loss_scaler = torch.amp.GradScaler("cuda")

    optim = SGD(params=model.parameters(), lr=0.001)

    pbar = tqdm(dataloader_training)
    for x, y in pbar:
        optim.zero_grad()
        x = x.to(training_config.device)
        y = y.to(training_config.device)
        if training_config.device == "cuda":
            x = x.half()  # I'm GPU-poor
            y = y.half()  # I'm GPU-poor
        pred = model(x)
        loss, loss_components = loss_func(pred, y)
        if loss_scaler:
            loss_scaler.scale(loss).backward()
        else:
            loss.backward()

        # for name, p in model.named_parameters():
        #     if p.grad is not None:
        #         val = p.grad.sum().detach().cpu().item()
        #     else:
        #         print(f"{name} has no grad")
        #         continue
        #     print(name, val)

        optim.step()
        box_loss, cls_loss, dfl_loss = loss_components
        pbar.set_description(f"loss: {loss.item()}  ")
