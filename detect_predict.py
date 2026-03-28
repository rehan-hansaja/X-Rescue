import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Set device for model inference (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Detection device:", device)

# Class mapping for fracture detection
# Index 0: Background/No Fracture, Index 1: Fracture
CLASS_NAMES = ["No Fracture", "Fracture"]

# Path to the pre-trained fracture detection model
DETECTION_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),  # Current script directory
    "models",  # Models folder
    "best_fracture_detection.pth"  # Model filename
)

def load_detection_model(model_path=DETECTION_MODEL_PATH):
    """
    Load a pre-trained Faster R-CNN model for fracture detection.
    Uses ResNet-101 as backbone with Feature Pyramid Network (FPN)
    for multi-scale feature extraction and object detection.
    Args:
        model_path (str): Path to the saved model checkpoint
    Returns:
        torch.nn.Module: Loaded Faster R-CNN model in evaluation mode
    """
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

    # BACKBONE CONFIGURATION
    # Create backbone network: ResNet-101 with Feature Pyramid Network
    # ResNet-101 provides deep feature extraction (101 layers)
    # FPN enables detection at multiple scales
    backbone = resnet_fpn_backbone("resnet101", weights=None)

    # MODEL INITIALIZATION
    # Initialize Faster R-CNN with the backbone and 2 classes:
    # - Class 0: Background (no fracture)
    # - Class 1: Fracture (target object to detect)
    model = FasterRCNN(backbone, num_classes=2)

    # LOAD CHECKPOINT WEIGHTS
    # Load saved model weights from disk
    state_dict = torch.load(model_path, map_location=device)

    # Handle DataParallel wrapper if model was trained with multiple GPUs
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Apply loaded weights to model
    model.load_state_dict(state_dict)

    # SET UP FOR INFERENCE
    # Move model to appropriate device
    model.to(device)
    # Set to evaluation mode (disables dropout, uses running stats for batch norm)
    model.eval()
    return model

def run_fracture_detection(model, input_path, score_threshold=0.45):
    """
    Run fracture detection on an input X-ray image.
    Workflow:
        1. Preprocess image (resize to 512x512, convert to tensor)
        2. Run inference through Faster R-CNN model
        3. Filter predictions by confidence score and class
        4. Return detection results with bounding boxes
    Args:
        model: Loaded Faster R-CNN model
        input_path (str): Path to input X-ray image
        score_threshold (float): Minimum confidence score for detections
    Returns:
        tuple: (label, confidence, boxes, scores)
            - label (str): "Fracture" or "No Fracture"
            - confidence (float): Maximum confidence score among detections
            - boxes (np.ndarray): Bounding boxes in xyxy format [x1, y1, x2, y2]
            - scores (np.ndarray): Confidence scores for each detection
    """

    # IMAGE PREPROCESSING
    # Define transformation pipeline
    transform = transforms.Compose([
        # Resize image to 512x512 pixels
        transforms.Resize((512, 512)),
        # Convert PIL image to PyTorch tensor and scale to [0, 1]
        transforms.ToTensor()
    ])

    # Load and convert image to RGB (Faster R-CNN expects 3-channel input)
    img_pil = Image.open(input_path).convert("RGB")
    # Apply transformations and move to the same device as model
    img_t = transform(img_pil).to(device)

    # MODEL INFERENCE
    # No gradient calculation needed for inference
    with torch.no_grad():
        # Model returns list of dicts (one dict per image)
        prediction = model([img_t])[0]

    # Extract detection results
    boxes = prediction["boxes"]  # Bounding boxes in xyxy format (N, 4)
    scores = prediction["scores"]  # Confidence scores (N,)
    labels = prediction["labels"]  # Class labels (N,)

    # FILTER PREDICTIONS
    # Condition: score > threshold AND class label is 1 (fracture)
    mask = (scores > score_threshold) & (labels == 1)
    boxes = boxes[mask]  # Filtered bounding boxes
    scores = scores[mask]  # Filtered confidence scores

    # IMAGE-LEVEL DECISION
    # Determine if fracture exists
    has_fracture = len(boxes) > 0
    label = "Fracture" if has_fracture else "No Fracture"
    # Maximum confidence among all detections
    max_conf = scores.max().item() if len(scores) > 0 else 0.0

    # RETURN RESULTS
    # Return all information needed for visualization and reporting
    return label, max_conf, boxes.cpu().numpy(), scores.cpu().numpy()