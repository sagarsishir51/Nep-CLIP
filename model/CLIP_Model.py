from transformers import AutoTokenizer, CLIPProcessor,CLIPModel
import torch
from tqdm.auto import tqdm

class CLIP_Model:
    def __init__(self,config):
        teacher_id = config["teacher_id"]
        self.model = CLIPModel.from_pretrained(teacher_id)
        self.processor = CLIPProcessor.from_pretrained(teacher_id)

    def get_text_features(self, texts, batch_size=256, device="cuda"):
        self.model.to(device).eval()
        text_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Text"):
                batch_texts = texts[i: i + batch_size]

                # Tokenize the raw strings — CLIP needs its own processor
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77  # CLIP's hard token limit
                ).to(device)

                # CLIPModel exposes get_text_features() directly

                text_f = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )

                if not isinstance(text_f, torch.Tensor):
                    text_f = text_f.pooler_output

                text_f = text_f / text_f.norm(dim=-1, keepdim=True)
                text_embs.append(text_f.cpu())

        return torch.cat(text_embs, dim=0).numpy()

    def get_image_features(self, image_values, batch_size=256, device="cuda"):
        img_embs = []
        self.model.to(device).eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(image_values), batch_size), desc="Encoding Images"):
                batch_pixels = image_values[i: i + batch_size].to(device)
                img_f = self.model.get_image_features(pixel_values=batch_pixels)
                # Handle potential pooler_output or raw output depending on CLIP version
                if hasattr(img_f, 'pooler_output'):
                    img_f = img_f.pooler_output

                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                img_embs.append(img_f.cpu())
        return torch.cat(img_embs, dim=0).numpy()  # [N_images, D]