from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, CLIPProcessor,CLIPModel
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F

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
        """
        Encodes text with self.model (raw text_model + text_projection). Supports two input shapes:

          - Flat list of strings:
              ["text1", "text2", ...] -> returns (N_texts, D)

          - Nested list (per-class templates, for ensembling):
              [["a photo of a dog", "a sketch of a dog", ...],   # class 0
               ["a photo of a cat", "a sketch of a cat", ...]]   # class 1
              -> averages + re-normalizes each group -> returns (N_classes, D)
        """
        self.model.to(device).eval()

        # --- 1. Detect shape, flatten if nested ---
        is_grouped = isinstance(texts[0], (list, tuple))

        if is_grouped:
            flat_texts, counts = [], []
            for group in texts:
                flat_texts.extend(group)
                counts.append(len(group))
        else:
            flat_texts, counts = texts, None

        text_embs = []

        # --- 2. Encode in batch_size chunks (unchanged from your original loop) ---
        with torch.no_grad():
            for i in tqdm(range(0, len(flat_texts), batch_size), desc=f"Encoding Text for {self.name}"):
                batch_texts = flat_texts[i: i + batch_size]

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

                text_f = text_outputs.pooler_output
                text_f = self.model.text_projection(text_f)
                text_f = text_f / text_f.norm(dim=-1, keepdim=True)
                text_embs.append(text_f.cpu())

        all_embs = torch.cat(text_embs, dim=0)  # (Total_texts, D)

        # --- 3. Flat input: return embeddings directly ---
        if not is_grouped:
            return all_embs  # (N_texts, D)

        # --- 4. Grouped input: average + re-normalize per class ---
        if len(set(counts)) == 1:
            n_templates, n_classes = counts[0], len(counts)
            grouped = all_embs.view(n_classes, n_templates, -1)
            class_embs = F.normalize(grouped.mean(dim=1), p=2, dim=-1)
        else:
            split_embs = torch.split(all_embs, counts, dim=0)
            class_embs = torch.stack([
                F.normalize(e.mean(dim=0), p=2, dim=0) for e in split_embs
            ], dim=0)

        return class_embs  # (N_classes, D)

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
        return torch.cat(img_embs, dim=0)  # [N_images, D]