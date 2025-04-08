# -*- coding: utf-8 -*-
import os

import os
from huggingface_hub import hf_hub_download

from . import model, processor, predictor
from .model import CLIP
from .processor import RuCLIPProcessor
from .predictor import Predictor

MODELS = {
    'ruclip-vit-base-patch32-224': dict(
        repo_id='ai-forever/ruclip-vit-base-patch32-224',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
    'ruclip-vit-base-patch16-224': dict(
        repo_id='ai-forever/ruclip-vit-base-patch16-224',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
    'ruclip-vit-large-patch14-224': dict(
        repo_id='ai-forever/ruclip-vit-large-patch14-224',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
    'ruclip-vit-large-patch14-336': dict(
        repo_id='ai-forever/ruclip-vit-large-patch14-336',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
    'ruclip-vit-base-patch32-384': dict(
        repo_id='ai-forever/ruclip-vit-base-patch32-384',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
    'ruclip-vit-base-patch16-384': dict(
        repo_id='ai-forever/ruclip-vit-base-patch16-384',
        filenames=[
            'bpe.model', 'config.json', 'pytorch_model.bin'
        ]
    ),
}

import os
from huggingface_hub import snapshot_download
from ruclip.model import CLIP
from ruclip.processor import RuCLIPProcessor

def load(name, device="cpu", cache_dir="/tmp/ruclip", use_auth_token=None):
    """
    Load a ruCLIP model by downloading the entire repository snapshot into a local folder,
    then loading the model from that folder.

    Parameters
    ----------
    name : str
        A model name listed in ruclip.MODELS.
    device : str or torch.device
        The device to load the model on (e.g. "cuda" or "cpu").
    cache_dir : str
        Base directory in which the repository snapshot will be cached.
    use_auth_token : Optional[str]
        An authentication token for the Hugging Face Hub (if required).

    Returns
    -------
    clip : torch.nn.Module
        The loaded ruCLIP model.
    clip_processor : RuCLIPProcessor
        The processor associated with the ruCLIP model.
    """
    from ruclip import MODELS
    assert name in MODELS, f"Available models: {list(MODELS.keys())}"
    config = MODELS[name]
    repo_id = config["repo_id"]

    # Define a target folder for the snapshot: e.g. /tmp/ruclip/ruclip-vit-base-patch32-384
    local_dir = os.path.join(cache_dir, name)
    os.makedirs(local_dir, exist_ok=True)

    # Download the entire repository snapshot into local_dir.
    # This will extract the repository contents so that config.json, pytorch_model.bin,
    # and other required files are at the root of local_dir.
    snapshot_folder = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        token=use_auth_token,
    )

    # Verify that the expected file is present.
    config_path = os.path.join(snapshot_folder, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Expected 'config.json' in {snapshot_folder} but found: {os.listdir(snapshot_folder)}"
        )

    # Load the model and processor from the snapshot folder.
    clip = CLIP.from_pretrained(snapshot_folder).eval().to(device)
    clip_processor = RuCLIPProcessor.from_pretrained(snapshot_folder)
    return clip, clip_processor

__all__ = ['processor', 'model', 'predictor', 'CLIP', 'RuCLIPProcessor', 'Predictor', 'MODELS', 'load']
__version__ = '0.0.2'
