import torch
import numpy as np
from numpy.typing import NDArray

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def nms(boxes: NDArray, scores: NDArray, iou_threshold: float = 0.5):
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


def process_predictions(
    pred: NDArray, confidence_threshold: float = 0.80, iou_threshold: float = 0.5
):
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
