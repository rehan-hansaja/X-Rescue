import torch
import cv2
import numpy as np
import os
from model_loader import load_swinir_model

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("SR device:", Device)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "swinir_enhanced_best_psnr.pth"
)

def load_model():
    model= load_swinir_model(MODEL_PATH)
    return model

def run_sr(model, input_path, output_path, tile=128, tile_overlap=32):
    device = next(model.parameters()).device

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # Resize before SR
    img = cv2.resize(img, (256, 256))

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    _, _, h, w = img.size()
    scale = 4

    output = torch.zeros(1, 3, h * scale, w * scale).to(device)
    weight = torch.zeros_like(output)

    model.eval()
    with torch.no_grad():
        for y in range(0, h, tile - tile_overlap):
            for x in range(0, w, tile - tile_overlap):
                y_end = min(y + tile, h)
                x_end = min(x + tile, w)

                patch = img[:, :, y:y_end, x:x_end]

                sr_patch = model(patch)

                oy, ox = y * scale, x * scale
                oy_end, ox_end = oy + sr_patch.size(2), ox + sr_patch.size(3)

                output[:, :, oy:oy_end, ox:ox_end] += sr_patch
                weight[:, :, oy:oy_end, ox:ox_end] += 1

    output /= weight
    output = output.clamp(0, 1)

    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, output)