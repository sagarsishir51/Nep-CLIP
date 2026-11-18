# core.py
"""Backend: model loading, index, encoders, and the two task functions."""
import os
import sys

# The UI lives in ui/; models, data, load_model, model/ and xai/ live in the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

from load_model import DISTILLED_MODELS_CONFIG
from model.NepClip_Model import NepClip_Model       # adjust path if different

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(ROOT, "Nep_CLIP_Models")
INDEX_PATH = os.path.join(ROOT, "index.npz")
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
)


def _build_configs(raw):
    if isinstance(raw, dict) and all(isinstance(v, dict) for v in raw.values()):
        return {v.get("name", k): v for k, v in raw.items()}
    if isinstance(raw, list) and all(isinstance(v, dict) for v in raw):
        return {v["name"]: v for v in raw}
    if isinstance(raw, dict) and "name" in raw:
        return {raw["name"]: raw}
    raise TypeError(f"Unrecognized DISTILLED_MODELS_CONFIG shape: {type(raw)}")

CONFIGS = _build_configs(DISTILLED_MODELS_CONFIG)
MODEL_NAMES = list(CONFIGS.keys())


# ---------------------------------------------------------------------------
# Shared index (image embeddings from the common CLIP image encoder)
# ---------------------------------------------------------------------------
INDEX_EMB, INDEX_PATHS = None, None
if os.path.exists(INDEX_PATH):
    _data = np.load(INDEX_PATH, allow_pickle=True)
    INDEX_EMB = _data["embeddings"].astype("float32")
    # ensure L2-normalized so dot product == cosine similarity
    _norms = np.linalg.norm(INDEX_EMB, axis=1, keepdims=True)
    _norms[_norms == 0] = 1e-8
    INDEX_EMB = INDEX_EMB / _norms
    INDEX_PATHS = list(_data["paths"])
    print(f"Loaded index: {len(INDEX_PATHS)} images. "
          f"mean norm before renorm: {float(_norms.mean()):.3f}")
else:
    print(f"No index found at {INDEX_PATH}.")


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_MODELS = {}   # name -> NepClip_Model


def get_model(name):
    if name not in _MODELS:
        print(f"Loading model '{name}'...")
        _MODELS[name] = NepClip_Model(config=CONFIGS[name], model_path=MODEL_PATH, device=DEVICE)
    return _MODELS[name]


def is_loaded(name):
    return name in _MODELS


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
def encode_text_query(model, text):
    feats = model.get_text_features([text], batch_size=1, device=DEVICE)
    return feats.numpy().astype("float32")[0]

def encode_texts(model, texts):
    feats = model.get_text_features(list(texts), batch_size=max(1, len(texts)), device=DEVICE)
    return feats.numpy().astype("float32")

def encode_image_query(model, img_path):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    pixel_values = model.processor(images=[img], return_tensors="pt")["pixel_values"]
    feats = model.get_image_features(pixel_values, batch_size=1, device=DEVICE)
    return feats.numpy().astype("float32")[0]


# ---------------------------------------------------------------------------
# Task 1: Image retrieval
# ---------------------------------------------------------------------------
def search(model_name, text_query, image_query, top_k):
    if INDEX_EMB is None:
        return [], f"No index at {INDEX_PATH}. Build it first."
    if not is_loaded(model_name):
        return [], f"Model '{model_name}' not loaded — click Load first."
    if not text_query and image_query is None:
        return [], "Enter a text query or upload an image to search."

    model = get_model(model_name)
    if image_query is not None:
        qvec = encode_image_query(model, image_query)
    else:
        qvec = encode_text_query(model, text_query)

    scores = INDEX_EMB @ qvec
    top = np.argsort(-scores)[: int(top_k)]
    gallery = [(INDEX_PATHS[i], f"{float(scores[i]):.3f}") for i in top]
    return gallery, f"Showing top {len(gallery)} results (model: {model_name})."


# ---------------------------------------------------------------------------
# Task 2: Zero-shot classification
# ---------------------------------------------------------------------------
def classify(model_name, image, labels_text):
    if not is_loaded(model_name):
        return {}, f"Model '{model_name}' not loaded — click Load first."
    if image is None:
        return {}, "Upload an image to classify."
    labels = [l.strip() for l in labels_text.splitlines() if l.strip()]
    if not labels:
        return {}, "Enter at least one candidate label (one per line)."

    model = get_model(model_name)
    img_vec = encode_image_query(model, image)
    label_vecs = encode_texts(model, labels)

    sims = label_vecs @ img_vec
    logits = sims * 100.0
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    result = {label: float(p) for label, p in zip(labels, probs)}
    top_label = max(result, key=result.get)
    return result, f"Predicted: '{top_label}' (model: {model_name})."



# --- add to core.py ---

def search_compare(name_a, name_b, mode, text_query, image_query, top_k):
    if INDEX_EMB is None:
        msg = f"No index at {INDEX_PATH}. Build it first."
        return [], [], msg, msg

    # pick query based on the active tab
    use_image = (mode == "image")
    if use_image and image_query is None:
        msg = "Upload an image in the Image query tab."
        return [], [], msg, msg
    if not use_image and not text_query:
        msg = "Enter text in the Text query tab."
        return [], [], msg, msg

    def one(name):
        if not is_loaded(name):
            return [], f"Model '{name}' not loaded — click Load first."
        model = get_model(name)
        if use_image:
            qvec = encode_image_query(model, image_query)
        else:
            qvec = encode_text_query(model, text_query)
        scores = INDEX_EMB @ qvec
        top = np.argsort(-scores)[: int(top_k)]
        gallery = [(INDEX_PATHS[i], f"{float(scores[i]):.3f}") for i in top]
        return gallery, f"Top {len(gallery)} (model: {name})."

    gal_a, stat_a = one(name_a)
    gal_b, stat_b = one(name_b)
    return gal_a, gal_b, stat_a, stat_b

def classify_compare(name_a, name_b, image, labels_text):
    """Zero-shot classify with two models; return two label dicts + two statuses."""
    if image is None:
        msg = "Upload an image to classify."
        return {}, {}, msg, msg
    labels = [l.strip() for l in labels_text.splitlines() if l.strip()]
    if not labels:
        msg = "Enter at least one candidate label (one per line)."
        return {}, {}, msg, msg

    def one(name):
        if not is_loaded(name):
            return {}, f"Model '{name}' not loaded — click Load first."
        model = get_model(name)
        img_vec = encode_image_query(model, image)
        label_vecs = encode_texts(model, labels)
        sims = label_vecs @ img_vec
        logits = sims * 100.0
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        result = {lb: float(p) for lb, p in zip(labels, probs)}
        top_label = max(result, key=result.get)
        return result, f"Predicted: '{top_label}' (model: {name})."

    res_a, stat_a = one(name_a)
    res_b, stat_b = one(name_b)
    return res_a, res_b, stat_a, stat_b

# ---------------------------------------------------------------------------
# Incremental indexing: encode new images and append to INDEX_PATH
# ---------------------------------------------------------------------------
def add_images_to_index(image_paths, model_name=None):
    """Encode new images with the shared image encoder and append to the index.
    Saves back to INDEX_PATH and updates the in-memory INDEX_EMB / INDEX_PATHS.
    Skips paths already present (by absolute path). Returns a status string."""
    global INDEX_EMB, INDEX_PATHS

    if not image_paths:
        return "No images selected."

    # need a model just for its shared image encoder + processor
    name = model_name or MODEL_NAMES[0]
    model = get_model(name)

    # normalize to absolute paths, skip ones already indexed
    existing = set(INDEX_PATHS) if INDEX_PATHS else set()
    new_paths = []
    for p in image_paths:
        ap = os.path.abspath(p)
        if ap not in existing and ap not in new_paths:
            new_paths.append(ap)

    if not new_paths:
        return "All selected images are already in the index."

    # encode each new image (reuses the shared teacher image encoder)
    new_vecs = []
    kept_paths = []
    for p in new_paths:
        try:
            v = encode_image_query(model, p)      # already L2-normalized
            new_vecs.append(v)
            kept_paths.append(p)
        except Exception as e:
            print(f"  skip (unreadable): {p} ({e})")

    if not new_vecs:
        return "None of the selected images could be read."

    new_emb = np.stack(new_vecs).astype("float32")   # (M, D)

    # append to in-memory index (or start fresh if none existed)
    if INDEX_EMB is None or len(INDEX_EMB) == 0:
        INDEX_EMB = new_emb
        INDEX_PATHS = list(kept_paths)
    else:
        # guard: dimensions must match
        if new_emb.shape[1] != INDEX_EMB.shape[1]:
            return (f"Dimension mismatch: new {new_emb.shape[1]} "
                    f"vs index {INDEX_EMB.shape[1]}. Aborted.")
        INDEX_EMB = np.concatenate([INDEX_EMB, new_emb], axis=0)
        INDEX_PATHS = INDEX_PATHS + list(kept_paths)

    # persist back to disk
    np.savez(INDEX_PATH, embeddings=INDEX_EMB, paths=np.array(INDEX_PATHS))

    return (f"Added {len(kept_paths)} image(s). "
            f"Index now has {len(INDEX_PATHS)} images.")

# --- add to core.py ---
def make_qr(data):
    """Return a PIL image QR code for the given text/URL, or None if empty."""
    if not data or not data.strip():
        return None
    import qrcode
    img = qrcode.make(data.strip())
    return img.get_image()      # PIL.Image for gr.Image

# ---------------------------------------------------------------------------
# Task 3: Explainability (word -> image region grounding, see xai/)
# ---------------------------------------------------------------------------
XAI_METHODS = ["gradcam++"]
XAI_THRESH = 0.5   # focus threshold for the box / contour overlays
DEVANAGARI_FONT = "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc"
_XAI_CTX = {}   # model_name -> run_analysis.Context


def _xai_context(model_name):
    from xai.nepclip_adapter import build_context_from_nepclip
    if model_name not in _XAI_CTX:
        _XAI_CTX[model_name] = build_context_from_nepclip(get_model(model_name), device=DEVICE)
    return _XAI_CTX[model_name]


def _center_square(img):
    """The region CLIPProcessor actually sees (shortest-side resize + center crop),
    so the heatmap lines up with the image it is drawn on."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def explain_terms(model_name, image, caption, terms, method, overlay_style, thresh=XAI_THRESH):
    """Ground each term (word or phrase) on the image. Returns (panel, gallery, status)."""
    import tempfile
    from PIL import Image
    from xai import attribution as A
    from xai.run_analysis import ground_words

    if not is_loaded(model_name):
        return None, [], f"Model '{model_name}' not loaded — click Load first."
    if image is None:
        return None, [], "No image to explain."
    if not os.path.exists(image):
        return None, [], f"Image file not found: {image}"
    if not terms:
        return None, [], "Nothing to explain."

    ctx = _xai_context(model_name)
    image_pil = _center_square(Image.open(image).convert("RGB"))
    cams = ground_words(ctx, image_pil, terms)

    out_dir = tempfile.mkdtemp(prefix="nepclip_xai_")
    font = DEVANAGARI_FONT if os.path.exists(DEVANAGARI_FONT) else None
    panel = A.render_word_panels(image_pil, terms, cams, out_dir, devanagari_font=font,
                                 overlay_style=overlay_style, thresh=float(thresh))
    gallery = [(os.path.join(out_dir, f"word_{A._safe(w)}.png"), w) for w in terms]
    return panel, gallery, f"Explained {len(terms)} term(s) with {method} (model: {model_name})."


def explain_compare(name_a, name_b, image, caption, terms, method, overlay_style,
                    thresh=XAI_THRESH):
    """XAI for the same image with two models. Returns (panel_a, panel_b, stat_a, stat_b)."""
    pa, _, sa = explain_terms(name_a, image, caption, terms, method, overlay_style, thresh)
    pb, _, sb = explain_terms(name_b, image, caption, terms, method, overlay_style, thresh)
    return pa, pb, sa, sb
