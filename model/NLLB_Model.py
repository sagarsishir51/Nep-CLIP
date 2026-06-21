from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, CLIPProcessor,CLIPModel
import torch
from tqdm.auto import tqdm

class NLLB_Model:
    def __init__(self,config):
        self.name = config["name"]
        student_id = config["id"]
        repo_id = config["repo_id"]
        teacher_id = config["teacher_id"]

        # Download the custom code locally to initialize the exact runtime architecture
        hf_hub_download(repo_id=repo_id, filename="configuration_nllb_clip.py", local_dir="..")
        hf_hub_download(repo_id=repo_id, filename="modeling_nllb_clip.py", local_dir="..")
        from modeling_nllb_clip import NLLBCLIPModel

        self.model = NLLBCLIPModel.from_pretrained(repo_id)
        self.processor = CLIPProcessor.from_pretrained(teacher_id)
        self.tokenizer = AutoTokenizer.from_pretrained(student_id)

    def get_text_features(self, texts, batch_size=256, device="cuda"):
        self.model.to(device).eval()
        text_embs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding Text for {self.name}"):
                batch_texts = texts[i: i + batch_size]

                # Tokenize the raw strings — CLIP needs its own processor
                text_inputs = self.tokenizer(
                    batch_texts,
                    padding="longest",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(device)

                text_outputs = self.model.text_model(
                    input_ids=text_inputs.input_ids,
                    attention_mask=text_inputs.attention_mask
                )
                # Extract the pooled matrix and project it down to 512 dim
                text_f = text_outputs.pooler_output
                text_f = self.model.text_projection(text_f)
                # L2-normalize so cosine sim == dot product
                text_f = text_f / text_f.norm(dim=-1, keepdim=True)
                text_embs.append(text_f.cpu())
        return torch.cat(text_embs, dim=0).numpy()

    def get_image_features(self, image_values, batch_size=256, device="cuda"):
        img_embs = []
        self.model.to(device).eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(image_values), batch_size), desc=f"Encoding Images for {self.name}"):
                batch_pixels = image_values[i: i + batch_size].to(device)
                img_f = self.model.get_image_features(pixel_values=batch_pixels)
                if hasattr(img_f, 'pooler_output'):
                    img_f = img_f.pooler_output

                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                img_embs.append(img_f.cpu())
        return torch.cat(img_embs, dim=0).numpy()  # [N_images, D]