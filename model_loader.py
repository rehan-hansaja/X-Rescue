import torch
from models.swinir import SwinIR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_swinir_model(model_path):
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=48,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
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
