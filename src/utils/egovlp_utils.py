import pathlib
import os, sys
import torch
import torch.nn as nn

EGOVLP_PATH = os.path.abspath("./EgoVLP-main")
if EGOVLP_PATH not in sys.path:
    sys.path.insert(0, EGOVLP_PATH)

#EgoVLP model 
from model.model import FrozenInTime

def load_egovlp(checkpoint_path: str, device: torch.device) -> nn.Module:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath #windows fix for posix

    original_dir = os.getcwd()
    os.chdir(EGOVLP_PATH)
    try:
        model_config = {
            "video_params": {
                "model": "SpaceTimeTransformer",
                "arch_config": "base_patch16_224",
                "num_frames": 16,
                "pretrained": True,
                "time_init": "zeros"
            },
            "text_params": {
                "model": "distilbert-base-uncased",
                "pretrained": True,
                "input": "text"
            },
            "projection": "minimal",
            "load_checkpoint": checkpoint_path
        }
        model = FrozenInTime(**model_config)
    finally:
        os.chdir(original_dir)
        pathlib.PosixPath = temp

    model.eval()
    model.to(device)
    return model