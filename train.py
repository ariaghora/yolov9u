import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from yolov9u.models import DetectionModel, ModelConfig


def nms(boxes, scores, iou_threshold=0.5):
    """
    Perform Non-Maximum Suppression on the bounding boxes.

    Args:
    - boxes: numpy array of shape (N, 4) where N is the number of boxes
             and each box is in format (x1, y1, x2, y2)
    - scores: numpy array of shape (N,) containing the confidence scores
    - iou_threshold: IoU threshold for considering a box as a duplicate

    Returns:
    - keep: numpy array of indices of the boxes to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


def process_predictions(pred, confidence_threshold=0.50, iou_threshold=0.9):
    """
    Process the raw predictions from YOLOv9.

    Args:
    - pred: numpy array of shape (8400, 84) where the first 4 columns are box coordinates
            (x_center, y_center, width, height) and the remaining 80 are class probabilities
    - image_size: tuple (height, width) of the original image
    - confidence_threshold: minimum confidence score to consider a detection
    - iou_threshold: IoU threshold for NMS

    Returns:
    - boxes: numpy array of shape (N, 4) where N is the number of final detections
    - labels: numpy array of shape (N,) containing the class labels
    - scores: numpy array of shape (N,) containing the confidence scores
    """

    # Extract box coordinates and class probabilities
    box_coords = pred[:, :4]
    class_probs = pred[:, 4:]

    # Get class scores and labels
    class_scores = np.max(class_probs, axis=1)
    class_labels = np.argmax(class_probs, axis=1)

    # Filter by confidence threshold
    mask = class_scores > confidence_threshold
    filtered_boxes = box_coords[mask]
    filtered_scores = class_scores[mask]
    filtered_labels = class_labels[mask]

    # Convert centroid format to corner format and scale to image size
    filtered_boxes[:, 0] = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
    filtered_boxes[:, 1] = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
    filtered_boxes[:, 2] = filtered_boxes[:, 0] + filtered_boxes[:, 2]
    filtered_boxes[:, 3] = filtered_boxes[:, 1] + filtered_boxes[:, 3]

    # Perform NMS
    keep = nms(filtered_boxes, filtered_scores, iou_threshold)

    return filtered_boxes[keep], filtered_labels[keep], filtered_scores[keep]


def draw_boxes(image, boxes, labels, scores, class_names):
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

    model = DetectionModel(config)
    model.load_state_dict(torch.load("./yolov9-e.pt", weights_only=True))
    model.eval()

    image = Image.open("./test.jpg")
    x = torch.FloatTensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        pred = model(x)[0]
    pred = pred[0].T.numpy()  # 8400x80

    filtered_boxes, filtered_labels, filtered_scores = process_predictions(pred)
    result_image = draw_boxes(
        image, filtered_boxes, filtered_labels, filtered_scores, list(range(80))
    )
    result_image.save("result.jpg")
