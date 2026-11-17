"""
run_analysis.py
===============
Word-to-region grounding for one image: one Grad-CAM++ map per word.
The Context is built by nepclip_adapter.build_context_from_nepclip.
"""


class Context:
    """What the grounding step needs from a loaded model."""
    def __init__(self, text_vector, preprocess, cam_fn):
        self.text_vector = text_vector   # fn(str) -> (1, D) normalized tensor
        self.preprocess = preprocess     # fn(PIL image) -> (3, 224, 224) tensor
        self.cam_fn = cam_fn             # fn(image_tensor, text_vector) -> (H, W) map in [0, 1]


def ground_words(ctx, image_pil, words):
    """Return one Grad-CAM++ map (H, W, float32 in [0, 1]) per word."""
    image_t = ctx.preprocess(image_pil).unsqueeze(0)
    cams = []
    for w in words:
        t = ctx.text_vector(w)                 # (1, D)
        gray = ctx.cam_fn(image_t, t)
        cams.append(gray.astype("float32"))
    return cams
