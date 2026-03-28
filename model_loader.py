import torch
from models.swinir import SwinIR

# Set device for model inference (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_swinir_model(model_path):
    """
    Load a pre-trained SwinIR model for image super-resolution.
    SwinIR (Swin Transformer for Image Restoration) is a state-of-the-art
    model for image super-resolution using Swin Transformer architecture.
    Args:
        model_path (str): Path to the saved model checkpoint file (.pth)
    Returns:
        SwinIR: Loaded model in evaluation mode, ready for inference
    """

    # MODEL ARCHITECTURE CONFIGURATION
    # Initialize SwinIR model with architecture parameters
    model = SwinIR(
        # Super-resolution scaling factor (4x upscaling)
        # Input: 64x64, Output: 256x256 after processing
        upscale=4,
        # Number of input channels (RGB image)
        in_chans=3,
        # Input image size for patch embedding
        img_size=48,
        # Window size for local self-attention in Swin Transformer
        window_size=16,
        # Range of input pixel values (0-1 normalized)
        img_range=1.,
        # Depths of each Swin Transformer block (6 stages)
        depths=[4, 4, 4, 4, 4, 4],
        # Embedding dimension
        embed_dim=180,
        # Number of attention heads for each stage
        num_heads=[4, 4, 4, 4, 4, 4],
        # MLP expansion ratio in transformer blocks
        mlp_ratio=2,
        # Upsampling method
        upsampler='pixelshuffle',
        # Residual connection type ('1conv' = single convolution)
        resi_connection='1conv'
    ).to(device)  # Move model to appropriate device

    # LOAD CHECKPOINT WEIGHTS
    # Load the saved model weights from disk
    state = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats:
    if 'params' in state:
        state = state['params']
    elif 'params_ema' in state:
        # EMA (Exponential Moving Average) weights (often better for inference)
        state = state['params_ema']

    # APPLY WEIGHTS TO MODEL
    # Load the state dictionary into the model
    # Any missing/unexpected keys will raise an error
    model.load_state_dict(state, strict=True)

    # SET TO EVALUATION MODE
    # - Disables dropout layers (if any)
    # - Disables batch norm's training-specific behavior
    # - Ensures deterministic inference
    model.eval()

    return model