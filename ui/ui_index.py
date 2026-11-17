import os
import sys

# The UI lives in ui/; models, data, load_model, model/ and xai/ live in the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import gradio as gr
import torch

# --- your existing model code ---
from load_model import DISTILLED_MODELS_CONFIG
from model.NepClip_Model import NepClip_Model       # adjust path if different

# ---------------------------------------------------------------------------
# Config / constants
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(ROOT, "Nep_CLIP_Models")
INDEX_PATH = os.path.join(ROOT, "index.npz")                      # single shared index (image embeddings)
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
)

# name -> config dict, for the dropdown (handles dict / list / single-dict shapes)
def _build_configs(raw):
    # dict keyed by name -> {"nepclip": {...}, ...}
    if isinstance(raw, dict) and all(isinstance(v, dict) for v in raw.values()):
        return {v.get("name", k): v for k, v in raw.items()}
    # list of config dicts -> [{...}, {...}]
    if isinstance(raw, list) and all(isinstance(v, dict) for v in raw):
        return {v["name"]: v for v in raw}
    # a single config dict -> {"name": ..., "teacher_id": ...}
    if isinstance(raw, dict) and "name" in raw:
        return {raw["name"]: raw}
    raise TypeError(f"Unrecognized DISTILLED_MODELS_CONFIG shape: {type(raw)}")

CONFIGS = _build_configs(DISTILLED_MODELS_CONFIG)
MODEL_NAMES = list(CONFIGS.keys())


# ---------------------------------------------------------------------------
# Single shared index (image embeddings from the common CLIP image encoder)
# ---------------------------------------------------------------------------
INDEX_EMB, INDEX_PATHS = None, None
if os.path.exists(INDEX_PATH):
    data = np.load(INDEX_PATH, allow_pickle=True)
    INDEX_EMB = data["embeddings"]
    INDEX_PATHS = list(data["paths"])
    print(f"Loaded index: {len(INDEX_PATHS)} images.")
else:
    print(f"No index found at {INDEX_PATH}.")

# Models are cached per name; the index is shared.
_MODELS = {}          # name -> NepClip_Model


def get_model(name):
    if name not in _MODELS:
        print(f"Loading model '{name}'...")
        _MODELS[name] = NepClip_Model(config=CONFIGS[name], model_path=MODEL_PATH, device=DEVICE)
    return _MODELS[name]


# ---------------------------------------------------------------------------
# Query encoders (use the selected model)
# ---------------------------------------------------------------------------
def encode_text_query(model, text):
    feats = model.get_text_features([text], batch_size=1, device=DEVICE)
    return feats.numpy().astype("float32")[0]

def encode_image_query(model, img_path):
    from PIL import Image
    img = Image.open(img_path).convert("RGB")
    pixel_values = model.processor(images=[img], return_tensors="pt")["pixel_values"]
    feats = model.get_image_features(pixel_values, batch_size=1, device=DEVICE)
    return feats.numpy().astype("float32")[0]


# ---------------------------------------------------------------------------
# Load button: two-step so the button shows a loading state
# ---------------------------------------------------------------------------
def start_loading(name):
    # immediate UI feedback before the (possibly slow) load runs
    return (
        gr.update(value="Loading…", interactive=False),   # load_btn
        f"⏳ Loading model '{name}'…",                      # model_status
        gr.update(interactive=False),                      # search_btn
    )


def load_selected_model(name):
    if INDEX_EMB is None:
        return (
            gr.update(value="Load", interactive=True),
            f"⚠️ No index at {INDEX_PATH} — build it first.",
            gr.update(interactive=False),
        )
    get_model(name)   # loads + caches (or reuses if already loaded)
    return (
        gr.update(value="Load", interactive=True),         # restore button
        f"✅ Model '{name}' loaded — index has {len(INDEX_PATHS)} images.",
        gr.update(interactive=True),                        # enable search
    )


# ---------------------------------------------------------------------------
# Dropdown change: disable Search until the new model is loaded
# ---------------------------------------------------------------------------
def on_model_change(name):
    return (
        f"Model '{name}' selected — click **Load** to use it.",
        gr.update(interactive=False),
    )


# ---------------------------------------------------------------------------
# Search with the loaded model + shared index
# ---------------------------------------------------------------------------
def search(model_name, text_query, image_query, top_k):
    if INDEX_EMB is None:
        return [], f"No index at {INDEX_PATH}. Build it first."
    if model_name not in _MODELS:
        return [], f"Model '{model_name}' not loaded — click Load first."
    if not text_query and image_query is None:
        return [], "Enter a text query or upload an image to search."

    model = _MODELS[model_name]
    if image_query is not None:
        qvec = encode_image_query(model, image_query)
    else:
        qvec = encode_text_query(model, text_query)

    scores = INDEX_EMB @ qvec
    top = np.argsort(-scores)[: int(top_k)]
    gallery = [(INDEX_PATHS[i], f"{float(scores[i]):.3f}") for i in top]
    return gallery, f"Showing top {len(gallery)} results (model: {model_name})."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="CLIP Image Retrieval", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 CLIP Image Retrieval")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=MODEL_NAMES,
            value=MODEL_NAMES[0],
            label="Model",
            scale=4,
        )
        load_btn = gr.Button("Load", variant="primary", scale=1)

    model_status = gr.Markdown(f"Model '{MODEL_NAMES[0]}' selected — click **Load** to use it.")

    gr.Markdown("### Search")
    with gr.Row():
        with gr.Column(scale=1):
            text_query = gr.Textbox(
                label="Text query",
                placeholder="e.g. 'a dog running on the beach'",
                lines=2,
            )
            gr.Markdown("**or**")
            image_query = gr.Image(label="Image query", type="filepath")
            top_k = gr.Slider(
                minimum=1, maximum=50, value=8, step=1, label="Number of results (top-k)"
            )
            search_btn = gr.Button("Search", variant="primary", interactive=False)  # off until loaded
            search_status = gr.Textbox(label="Search status", interactive=False)

        with gr.Column(scale=2):
            results_gallery = gr.Gallery(
                label="Results",
                columns=4,
                height="auto",
                object_fit="cover",
            )

    # Load button -> load model, enable Search
    load_btn.click(
        fn=start_loading,
        inputs=[model_dropdown],
        outputs=[load_btn, model_status, search_btn],
    ).then(
        fn=load_selected_model,
        inputs=[model_dropdown],
        outputs=[load_btn, model_status, search_btn],
    )
    # Switching model -> disable Search until re-loaded
    model_dropdown.change(
        fn=on_model_change,
        inputs=[model_dropdown],
        outputs=[model_status, search_btn],
    )

    search_btn.click(
        fn=search,
        inputs=[model_dropdown, text_query, image_query, top_k],
        outputs=[results_gallery, search_status],
    )


if __name__ == "__main__":
    demo.launch()