from typing import List

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from yolov9u.models import DetectionModel, ModelConfig
from yolov9u.postprocessing import process_predictions


def draw_boxes(
    image: Image.Image,
    boxes: NDArray,
    labels: NDArray,
    scores: NDArray,
    class_names: List[str | int],
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        class_name = class_names[label]
        label_text = f"{class_name}: {score:.2f}"

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw label
        text_bbox = font.getbbox(label_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1, x1 + text_width, y1 + text_height], fill="red")
        draw.text((x1, y1), label_text, fill="white", font=font)

    return image


if __name__ == "__main__":
    with open("./config/yolov9-gelan-e.yaml") as f:
        config = ModelConfig(**yaml.load(f, yaml.SafeLoader))

    model = DetectionModel(config).float()
    model.load_state_dict(torch.load("./yolov9-e.pt", weights_only=True))
    model.eval()

    image = Image.open("./assets/images/car.jpg")
    image = image.resize((640, 640))
    x = torch.FloatTensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        pred = model(x)[0]
    pred = pred[0].T.numpy()

    filtered_boxes, filtered_labels, filtered_scores = process_predictions(pred)
    result_image = draw_boxes(
        image, filtered_boxes, filtered_labels, filtered_scores, list(range(80))
    )
    result_image.save("./assets/images/result.jpg")
