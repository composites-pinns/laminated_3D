import torch


def get_device() -> torch.device:
    """Select best available device: CUDA if available, otherwise CPU.

    Apple MPS is explicitly ignored because this project requires float64
    precision which MPS does not fully support. This helper therefore never
    returns an 'mps' device.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
