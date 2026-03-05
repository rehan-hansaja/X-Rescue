# detect_predict.py
import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Detection device:", device)

CLASS_NAMES = ["No Fracture", "Fracture"]   # index 0 = background, 1 = fracture

DETECTION_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best_fracture_detection.pth"
)

def load_detection_model(model_path=DETECTION_MODEL_PATH):
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

    backbone = resnet_fpn_backbone("resnet101", weights=None)
    model = FasterRCNN(backbone, num_classes=2)  # 0=background, 1=fracture

    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_fracture_detection(model, input_path, score_threshold=0.45):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    img_pil = Image.open(input_path).convert("RGB")
    img_t = transform(img_pil).to(device)

    with torch.no_grad():
        prediction = model([img_t])[0]   # list of dict → take [0]

    boxes  = prediction["boxes"]         # (N,4)  xyxy format
    scores = prediction["scores"]        # (N,)
    labels = prediction["labels"]        # (N,)   should be 1 for fracture

    # Filter high-confidence fracture predictions
    mask = (scores > score_threshold) & (labels == 1)
    boxes  = boxes[mask]
    scores = scores[mask]

    # Image-level decision
    has_fracture = len(boxes) > 0
    label = "Fracture" if has_fracture else "No Fracture"
    max_conf = scores.max().item() if len(scores) > 0 else 0.0

    # Return boxes & scores so app.py can draw them
    return label, max_conf, boxes.cpu().numpy(), scores.cpu().numpy()