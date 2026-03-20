import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.swinir import SwinIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Models

def load_swinir_model(model_path, depths, num_heads, window_size):
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=48,
        window_size=window_size,
        img_range=1.,
        depths=depths,
        embed_dim=180,
        num_heads=num_heads,
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    ).to(device)

    state = torch.load(model_path, map_location=device)
    if 'params' in state:
        state = state['params']
    elif 'params_ema' in state:
        state = state['params_ema']

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# Update these paths to your actual .pth files
ENHANCED_MODEL_PATH = r"C:\LOCAL DISK\CS\projects\X-Rescue\models\swinir_enhanced_best_psnr.pth"
BASIC_MODEL_PATH    = r"C:\LOCAL DISK\CS\projects\X-Rescue\models\swinir_best_psnr.pth"

print("Loading Enhanced SwinIR (4 STL, num_heads=4, window=16)...")
enhanced_model = load_swinir_model(
    ENHANCED_MODEL_PATH,
    depths=[4, 4, 4, 4, 4, 4],
    num_heads=[4, 4, 4, 4, 4, 4],
    window_size=16
)

print("Loading Basic SwinIR (6 STL, num_heads=6, window=8)...")
basic_model = load_swinir_model(
    BASIC_MODEL_PATH,
    depths=[6, 6, 6, 6, 6, 6],
    num_heads=[6, 6, 6, 6, 6, 6],
    window_size=8
)

print("Both models loaded successfully.\n")

# Load & Preprocess Input Image

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),   # simulate LR input
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 64, 64]


# Update this to your test image path
IMAGE_PATH = r"C:\LOCAL DISK\CS\projects\X-Rescue\images\uploads\LR_IMG0003703.png"

input_tensor = preprocess(IMAGE_PATH)

# Benchmark Function

def benchmark(model, input_tensor, model_name, runs=50):
    print(f"Benchmarking {model_name} over {runs} runs...")

    # Warm-up (3 runs before timing)
    for _ in range(3):
        with torch.no_grad():
            model(input_tensor)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            output = model(input_tensor)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # convert to ms

    avg   = np.mean(times)
    best  = np.min(times)
    worst = np.max(times)
    std   = np.std(times)

    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  Average inference time  : {avg:.2f} ms")
    print(f"  Best    inference time  : {best:.2f} ms")
    print(f"  Worst   inference time  : {worst:.2f} ms")
    print(f"  Std deviation           : {std:.2f} ms")
    print(f"  Output shape            : {output.shape}")

    return avg, output


# Run Benchmark

print(f"Device       : {device}")
print(f"Input shape  : {input_tensor.shape}")
print()

avg_basic,    output_basic    = benchmark(basic_model,    input_tensor, "Basic SwinIR    (6 STL | num_heads=6 | window=8)")
avg_enhanced, output_enhanced = benchmark(enhanced_model, input_tensor, "Enhanced SwinIR (4 STL | num_heads=4 | window=16)")

# Summary

improvement = ((avg_basic - avg_enhanced) / avg_basic) * 100

print(f"\n{'═'*50}")
print(f"  SUMMARY")
print(f"{'═'*50}")
print(f"  Basic SwinIR    (6 STL) : {avg_basic:.2f} ms")
print(f"  Enhanced SwinIR (4 STL) : {avg_enhanced:.2f} ms")
print(f"  Speed improvement       : {improvement:.1f}% faster")
print(f"{'═'*50}\n")

# Save Output Images

def save_output(tensor, filename):
    tensor = tensor.squeeze(0).clamp(0, 1).cpu()
    img = transforms.ToPILImage()(tensor)
    img.save(filename)
    print(f"Saved: {filename}")

save_output(output_basic,    "output_basic_swinir.png")
save_output(output_enhanced, "output_enhanced_swinir.png")
print("\nDone! Compare output_basic_swinir.png and output_enhanced_swinir.png for visual quality check.")