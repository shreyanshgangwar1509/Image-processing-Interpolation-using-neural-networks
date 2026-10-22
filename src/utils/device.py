"""Device selection utilities."""
import torch


def get_device(name: str = "auto") -> torch.device:
    """Returns a torch.device based on requested string ('auto', 'cpu', 'cuda', 'cuda:0', etc.)."""
    if name == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = name
    return torch.device(device_str)
