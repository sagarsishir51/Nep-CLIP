"""
attribution.py — render word-to-region maps for Nep-CLIP
========================================================
Draws each word's map over the image as a heatmap, a box, or a contour, and
saves one PNG per word plus a combined panel.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


def upsample_map(cam_7x7, size):
    """Bilinearly upsample a small (g x g) map to (W, H) using PIL."""
    m = cam_7x7 - cam_7x7.min()
    m = m / (m.max() + 1e-9)
    img = Image.fromarray((m * 255).astype("uint8"))
    img = img.resize(size, resample=Image.BILINEAR)
    return np.asarray(img).astype("float32") / 255.0


def overlay(image_pil, cam_full, alpha=0.45):
    """Blend a [0,1] heatmap over the RGB image with the 'jet' colormap."""
    base = np.asarray(image_pil.convert("RGB")).astype("float32") / 255.0
    try:
        jet = matplotlib.colormaps["jet"]           # matplotlib >= 3.7
    except AttributeError:
        jet = cm.get_cmap("jet")                     # older matplotlib
    heat = jet(cam_full)[..., :3]                    # RGBA -> RGB
    return (1 - alpha) * base + alpha * heat


def overlay_box(image_pil, cam_full, thresh=0.5):
    """Draw a rectangle around the largest above-threshold region of cam_full.
    Returns (rgb_float_image, bbox) where bbox is (x0,y0,x1,y1) or None."""
    base = np.asarray(image_pil.convert("RGB")).astype("float32") / 255.0
    mask = cam_full >= thresh
    if not mask.any():                               # fall back to the peak cell
        yx = np.unravel_index(np.argmax(cam_full), cam_full.shape)
        mask = np.zeros_like(cam_full, dtype=bool)
        mask[yx] = True
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    # draw a red rectangle border (a few px thick) on a copy
    out = base.copy()
    t = max(2, int(0.006 * max(out.shape[:2])))
    red = np.array([1.0, 0.0, 0.0], dtype="float32")
    for (ya, yb, xa, xb) in [(y0, y0 + t, x0, x1), (y1 - t, y1, x0, x1),
                             (y0, y1, x0, x0 + t), (y0, y1, x1 - t, x1)]:
        out[ya:yb, xa:xb] = red
    return out, (int(x0), int(y0), int(x1), int(y1))


def overlay_contour(image_pil, cam_full, levels=(0.5,)):
    """Return (rgb_float_image, contour_fn) — the image plus a closure that draws
    contour lines on a matplotlib axis (matplotlib handles the contouring)."""
    base = np.asarray(image_pil.convert("RGB")).astype("float32") / 255.0
    def draw(ax):
        ax.contour(cam_full, levels=list(levels), colors="red", linewidths=1.5)
    return base, draw


def render_word_panels(image_pil, words, cams, out_dir, devanagari_font=None,
                       overlay_style="heatmap", thresh=0.5):
    """Save one overlay per word + a combined row panel.
    overlay_style: 'heatmap' (soft jet), 'box' (rectangle round focus region),
                   or 'contour' (outline at the threshold level)."""
    os.makedirs(out_dir, exist_ok=True)
    W, H = image_pil.size

    fontprops = None
    if devanagari_font and os.path.exists(devanagari_font):
        from matplotlib import font_manager
        fontprops = font_manager.FontProperties(fname=devanagari_font)

    def make_axis_image(cam):
        """Return (rgb_image, optional_draw_fn) for one word's map, per style."""
        cam_full = upsample_map(cam, (W, H))
        if overlay_style == "box":
            img, _ = overlay_box(image_pil, cam_full, thresh=thresh)
            return img, None
        if overlay_style == "contour":
            img, draw = overlay_contour(image_pil, cam_full, levels=(thresh,))
            return img, draw
        return overlay(image_pil, cam_full), None      # heatmap (default)

    per_word = []
    for w, cam in zip(words, cams):
        img, draw = make_axis_image(cam)
        per_word.append((img, draw))
        plt.figure(figsize=(4, 4))
        ax = plt.gca()
        ax.imshow(img)
        if draw:
            draw(ax)
        ax.axis("off")
        ax.set_title(w, fontproperties=fontprops)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"word_{_safe(w)}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()

    n = len(words) + 1
    plt.figure(figsize=(3 * n, 3.2))
    ax = plt.subplot(1, n, 1)
    ax.imshow(np.asarray(image_pil.convert("RGB")))
    ax.set_title("image"); ax.axis("off")
    for i, (w, (img, draw)) in enumerate(zip(words, per_word), start=2):
        ax = plt.subplot(1, n, i)
        ax.imshow(img)
        if draw:
            draw(ax)
        ax.set_title(w, fontproperties=fontprops)
        ax.axis("off")
    panel = os.path.join(out_dir, "panel_all_words.png")
    plt.tight_layout()
    plt.savefig(panel, dpi=200, bbox_inches="tight")
    plt.close()
    return panel


def _safe(s):
    return "".join(c if c.isalnum() else "_" for c in s)[:24]
