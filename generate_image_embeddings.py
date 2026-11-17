# clip_index.py
"""
Build and persist image embeddings using NepClip_Model, so you don't
re-encode every run.
"""

import os
import glob
import json
import argparse
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from load_model import DISTILLED_MODELS_CONFIG
from model.NepClip_Model import NepClip_Model   # adjust import to your path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def pick_device(prefer=None):
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Model loading (cached so repeated calls in one run are cheap)
# ---------------------------------------------------------------------------
_model = None


def load_model(config, model_path, device=None):
    """`config` is a dict (or path to a JSON file) with keys:
        name, teacher_id, id, file_name
    `model_path` is the folder containing the checkpoint (config['file_name']).
    """
    global _model
    if _model is None:
        if isinstance(config, str):
            with open(config) as f:
                config = json.load(f)
        device = pick_device(device)
        _model = NepClip_Model(config=config, model_path=model_path, device=device)
        _model._device = device            # stash for later calls
    return _model


# ---------------------------------------------------------------------------
# Gather image files from dirs / globs / single files
# ---------------------------------------------------------------------------
def collect_image_paths(paths):
    found = []
    for p in paths:
        if os.path.isdir(p):
            for ext in IMAGE_EXTS:
                found += glob.glob(os.path.join(p, "**", f"*{ext}"), recursive=True)
                found += glob.glob(os.path.join(p, "**", f"*{ext.upper()}"), recursive=True)
        elif os.path.isfile(p):
            found.append(p)
        else:
            found += glob.glob(p, recursive=True)
    return sorted(set(os.path.abspath(f) for f in found))


# ---------------------------------------------------------------------------
# Encode images -> normalized embeddings, save to disk
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_index(model, paths, out_path="index.npz", chunk_size=256):
    """Preprocess + encode images in chunks so RAM stays bounded on big sets."""
    device = getattr(model, "_device", "cpu")
    files = collect_image_paths(paths)
    if not files:
        raise ValueError("No images found in the given paths.")

    all_embs, valid_paths = [], []

    for start in range(0, len(files), chunk_size):
        chunk = files[start:start + chunk_size]
        pil_imgs, chunk_paths = [], []
        for f in chunk:
            try:
                pil_imgs.append(Image.open(f).convert("RGB"))
                chunk_paths.append(f)
            except Exception as e:
                print(f"  skip (unreadable): {f} ({e})")
        if not pil_imgs:
            continue

        # processor -> pixel_values tensor (N, 3, H, W)
        pixel_values = model.processor(images=pil_imgs, return_tensors="pt")["pixel_values"]

        # NepClip_Model.get_image_features returns normalized embeddings on CPU
        embs = model.get_image_features(pixel_values, batch_size=chunk_size, device=device)
        all_embs.append(embs)
        valid_paths.extend(chunk_paths)
        print(f"  encoded {len(valid_paths)}/{len(files)}")

    emb = torch.cat(all_embs, dim=0).numpy().astype("float32")
    np.savez(
        out_path,
        embeddings=emb,
        paths=np.array(valid_paths),
        model_name=model.name,
    )
    print(f"Saved {len(valid_paths)} embeddings -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Load a saved index
# ---------------------------------------------------------------------------
@dataclass
class Index:
    embeddings: np.ndarray   # (N, D), L2-normalized
    paths: list


def load_index(path="index.npz"):
    data = np.load(path, allow_pickle=True)
    return Index(embeddings=data["embeddings"], paths=list(data["paths"]))


# ---------------------------------------------------------------------------
# Query encoders + search
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_text(model, text):
    """Nepali/English text query -> normalized vector via student encoder."""
    device = getattr(model, "_device", "cpu")
    feats = model.get_text_features([text], batch_size=1, device=device)  # (1, D)
    return feats.numpy().astype("float32")[0]


@torch.no_grad()
def encode_image(model, img):
    """Example-image query -> normalized vector via teacher image encoder.
    `img` is a PIL.Image or a path."""
    device = getattr(model, "_device", "cpu")
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    pixel_values = model.processor(images=[img], return_tensors="pt")["pixel_values"]
    feats = model.get_image_features(pixel_values, batch_size=1, device=device)  # (1, D)
    return feats.numpy().astype("float32")[0]


def search(query_vec, index: Index, top_k=8):
    scores = index.embeddings @ query_vec          # cosine sim (all normalized)
    top = np.argsort(-scores)[:top_k]
    return [(index.paths[i], float(scores[i])) for i in top]


# ---------------------------------------------------------------------------
# Defaults — edit these instead of passing CLI args every run
# ---------------------------------------------------------------------------

DEFAULT_PATHS = [
    # 'dataset/flickr8k/images/Flicker8k_Dataset',
    # 'dataset/flickr30k/images',
    # 'dataset/xm3600/images',
    # 'dataset/xtd10'
    'dataset/nepali_dataset'
]
DEFAULT_OUT = "index.npz"
DEFAULT_CONFIG = DISTILLED_MODELS_CONFIG["BERT"]
DEFAULT_MODEL_PATH = "Nep_CLIP_Models"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_DEVICE = "mps"            # None -> auto (cuda / mps / cpu)

# ---------------------------------------------------------------------------
# CLI (all args optional — defaults come from the constants above)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a NepClip image index.")
    parser.add_argument("--paths", nargs="+", default=DEFAULT_PATHS,
                        help="Directories, files, or glob patterns.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="JSON with name/teacher_id/id/file_name.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="Folder containing the checkpoint file.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()

    model = load_model(args.config, args.model_path, args.device)
    build_index(model, args.paths, args.out, args.chunk_size)