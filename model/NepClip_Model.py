from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, CLIPProcessor,CLIPModel
import torch
from tqdm import tqdm
from model.NepClipEncoder import NepClipEncoder
import torch.nn.functional as F


class NepClip_Model:
    def __init__(self,config,device):
        self.name = config["name"]
        teacher_id = config["teacher_id"]
        student_id = config["id"]
        self.processor = CLIPProcessor.from_pretrained(teacher_id)
        self.model = CLIPModel.from_pretrained(teacher_id)
        best_kw = {'do_lower_case': False, 'strip_accents': False}
        self.tokenizer = AutoTokenizer.from_pretrained(student_id, **best_kw)
        self.nep_clip_encoder = self.load_nep_clip_encoder(config=config, device=device)

    def load_nep_clip_encoder(self, config, device):
        student_id = config["id"]
        ckpt_name = f'{config["hf_repo"]}/{config["hf_file"]}'

        # 1. Download the checkpoint from Hugging Face (cached after the first run)
        print(f"Loading weights from: {ckpt_name}")
        ckpt_path = hf_hub_download(repo_id=config["hf_repo"], filename=config["hf_file"])

        # 2. Initialize a fresh instance of the encoder using the HuggingFace ID
        nep_clip_encoder = NepClipEncoder(student_id)

        # 3. Load the checkpoint to the target device safely
        ckpt = torch.load(ckpt_path, map_location=device)

        # 4. Load state dicts into the sub-modules
        if isinstance(ckpt, dict) and "student_state_dict" in ckpt and "projection_state_dict" in ckpt:
            # Custom dictionary tracking structure
            nep_clip_encoder.student.load_state_dict(ckpt["student_state_dict"])
            nep_clip_encoder.projection.load_state_dict(ckpt["projection_state_dict"])
        else:
            # Fallback: If the checkpoint itself IS the raw state dict
            try:
                nep_clip_encoder.load_state_dict(ckpt)
            except Exception as e:
                print(f" ⚠️ Could not automatically parse weights for {ckpt_name}. Error: {e}")

        # 5. Send the entire encoder model to the selected device (MPS/CUDA/CPU)
        nep_clip_encoder.to(device)

        # Set to evaluation mode
        nep_clip_encoder.eval()
        return nep_clip_encoder

    def get_text_features(self, texts, batch_size, device):
        """
        Encodes text with nep_clip_encoder. Supports two input shapes:

          - Flat list of strings:
              ["text1", "text2", ...] -> returns (N_texts, D)

          - Nested list (per-class templates, for ensembling):
              [["a photo of dog", "a sketch of dog", ...],   # class 0
               ["a photo of cat", "a sketch of cat", ...]]   # class 1
              -> averages + re-normalizes each group -> returns (N_classes, D)
        """
        self.nep_clip_encoder.to(device).eval()

        # --- 1. Detect shape, flatten if nested ---
        is_grouped = isinstance(texts[0], (list, tuple))

        if is_grouped:
            flat_texts, counts = [], []
            for group in texts:
                flat_texts.extend(group)
                counts.append(len(group))
        else:
            flat_texts, counts = texts, None

        # --- 2. Tokenize everything once ---
        tokenized_inputs = self.tokenizer(
            flat_texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        num_texts = tokenized_inputs['input_ids'].shape[0]
        all_embs = []

        # --- 3. Encode in batch_size chunks regardless of grouping ---
        with torch.no_grad():
            for i in tqdm(range(0, num_texts, batch_size), desc=f"Encoding Text for {self.name}"):
                batch_text = {k: v[i: i + batch_size].to(device) for k, v in tokenized_inputs.items()}
                txt_f = self.nep_clip_encoder(batch_text)
                txt_f = F.normalize(txt_f, p=2, dim=-1)
                all_embs.append(txt_f.cpu())

        all_embs = torch.cat(all_embs, dim=0)  # (Total_texts, D)

        # --- 4. Flat input: return embeddings directly ---
        if not is_grouped:
            return all_embs  # (N_texts, D)

        # --- 5. Grouped input: average + re-normalize per class ---
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
                # Handle potential pooler_output or raw output depending on CLIP version
                if hasattr(img_f, 'pooler_output'):
                    img_f = img_f.pooler_output

                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                img_embs.append(img_f.cpu())
        return torch.cat(img_embs, dim=0)  # [N_images, D]