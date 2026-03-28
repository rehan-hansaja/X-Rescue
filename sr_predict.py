import torch
import cv2
import numpy as np
import os
from model_loader import load_swinir_model

# Set device for inference (GPU if available, otherwise CPU)
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("SR device:", Device)

# Path to the pre-trained SwinIR model weights
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),  # Current script directory
    "models",  # Models folder
    "swinir_enhanced_best_psnr.pth"  # Model filename
)

def load_model():
    """
    Load the SwinIR super-resolution model from the saved checkpoint.
    """
    # Load model architecture with pre-trained weights
    model = load_swinir_model(MODEL_PATH)
    # Set to evaluation mode (disables dropout, batch norm behavior changes)
    model.eval()
    return model


def run_sr(model, input_path, output_path):
    """
    Perform super-resolution enhancement on an input X-ray image.
    Workflow:
        1. Load and preprocess input image
        2. Resize to model input size (64x64)
        3. Run inference through SwinIR model
        4. Post-process output and blend with original
        5. Save enhanced image

    Args:
        model: Loaded SwinIR model instance
        input_path (str): Path to input low-resolution image
        output_path (str): Path where enhanced image will be saved
    """
    # Determine the device the model is on (CPU or GPU)
    device = next(model.parameters()).device

    # LOAD AND PREPROCESS IMAGE
    # Read image in BGR format
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    # Store original dimensions for final resizing
    original_h, original_w = img.shape[:2]

    # Convert BGR to RGB (model expects RGB format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values from [0, 255] to [0, 1]
    img = img.astype(np.float32) / 255.0
    # Resize to 64x64 (SwinIR's expected input size for this model)
    # Using bicubic interpolation for smoother downsampling
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

    # Convert numpy array to PyTorch tensor
    # Permute from (H, W, C) to (C, H, W) for PyTorch convention
    img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0).to(device)

    # MODEL INFERENCE
    # Use inference_mode() for faster inference and reduced memory usage
    with torch.inference_mode():
        output = model(img)  # Forward pass through SwinIR model

    # POST-PROCESS OUTPUT
    # Clamp values to valid range [0, 1] to avoid artifacts
    output = output.clamp(0, 1)
    # Remove batch dimension and convert back to (H, W, C) format
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Convert back to [0, 255] range and uint8 type
    output = (output * 255).astype(np.uint8)
    # Convert RGB back to BGR for OpenCV compatibility
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # Resize SR output back to original input dimensions
    # Using Lanczos interpolation for high-quality upscaling
    output = cv2.resize(output, (original_w, original_h), interpolation=cv2.INTER_LANCZOS4)

    # BLEND WITH ORIGINAL
    # Blend SR output with original image to preserve fracture details
    # This prevents over-smoothing that might lose important diagnostic features
    input_bgr = cv2.imread(input_path)  # Reload original (already in BGR)
    # Weighted blend: 60% SR, 40% original
    # Adjustable ratio - higher SR weight = more enhancement, lower = more original detail
    output = cv2.addWeighted(output, 0.6, input_bgr, 0.4, 0)

    # SAVE RESULT
    cv2.imwrite(output_path, output)