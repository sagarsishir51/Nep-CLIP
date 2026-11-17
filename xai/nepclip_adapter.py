# nepclip_adapter.py
# ---------------------------------------------------------------------------
# Wrap a loaded NepClip_Model for Grad-CAM++ (pytorch_grad_cam, pip install grad-cam).
#
#   - CLIPImageWrapper : forward(pixel_values) -> image embedding
#   - SimilarityToText : CAM target = cosine(image_emb, word_text_vector)
#   - reshape_transform: ViT patch tokens (drop CLS, 49 -> 7x7)
# ---------------------------------------------------------------------------
import torch.nn as nn

from xai.run_analysis import Context


class CLIPImageWrapper(nn.Module):
    """forward(pixel_values) -> image embedding (B, D). Lets pytorch_grad_cam
    treat the CLIP image tower like a normal model whose 'output' is the embedding."""
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, pixel_values):
        v = self.clip.get_image_features(pixel_values=pixel_values)
        if hasattr(v, "pooler_output"):
            v = v.pooler_output
        return v


class SimilarityToText:
    """CAM target = cosine similarity between the image embedding and a fixed,
    normalized text vector t. One instance per word."""
    def __init__(self, t):
        # t: (D,) tensor on the model's device, already normalized
        self.t = t

    def __call__(self, model_output):
        v = model_output
        if v.dim() == 1:
            v = v.unsqueeze(0)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-9)
        return (v * self.t).sum()


def _reshape_transform(tensor, height=7, width=7):
    """(B, seq=1+H*W, dim) -> drop CLS -> (B, dim, H, W) for pytorch_grad_cam."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def build_context_from_nepclip(nepclip_model, device="cuda"):
    """Return a Context whose cam_fn runs Grad-CAM++ for one word vector."""
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
    except ImportError:
        raise ImportError("pytorch_grad_cam not installed. Run: pip install grad-cam")

    m = nepclip_model
    m.model.to(device).eval()
    m.nep_clip_encoder.to(device).eval()

    def text_vector(text):
        emb = m.get_text_features([text], batch_size=1, device=device)  # (1,D), normalized, CPU
        return emb.to(device)

    def preprocess(pil_image):
        px = m.processor(images=pil_image, return_tensors="pt")["pixel_values"][0]
        return px.to(device)

    wrapper = CLIPImageWrapper(m.model).to(device).eval()
    # target layer: last vision block's first layer-norm (its patch tokens still
    # receive gradient; the block's own patch outputs do not)
    last_block = m.model.vision_model.encoder.layers[-1]
    target_layers = [last_block.layer_norm1]

    def cam_fn(image_tensor, t_vec):
        t = t_vec.squeeze(0) if t_vec.dim() == 2 else t_vec
        t = (t / (t.norm() + 1e-9)).to(device)
        with GradCAMPlusPlus(model=wrapper, target_layers=target_layers,
                             reshape_transform=_reshape_transform) as cam_obj:
            grayscale = cam_obj(input_tensor=image_tensor.to(device),
                                targets=[SimilarityToText(t)])
        return grayscale[0]

    return Context(text_vector=text_vector, preprocess=preprocess, cam_fn=cam_fn)
