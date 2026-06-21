import os
from transformers import AutoTokenizer, CLIPProcessor,CLIPModel
import torch
from tqdm import tqdm
from model.NepClipEncoder import NepClipEncoder


class NepClip_Model:
    def __init__(self,config,model_path,device):
        self.name = config["name"]
        teacher_id = config["teacher_id"]
        student_id = config["id"]
        self.processor = CLIPProcessor.from_pretrained(teacher_id)
        self.model = CLIPModel.from_pretrained(teacher_id)
        best_kw = {'do_lower_case': False, 'strip_accents': False}
        self.tokenizer = AutoTokenizer.from_pretrained(student_id, **best_kw)
        self.nep_clip_encoder = self.load_nep_clip_encoder(model_folder=model_path, config=config, device=device)

    def load_nep_clip_encoder(self,model_folder, config, device):
        student_id = config["id"]
        ckpt_name = config["file_name"]

        print(f"Loading weights from: {ckpt_name}")
        ckpt_path = os.path.join(model_folder, ckpt_name)

        # Check if the file actually exists before processing
        if not os.path.exists(ckpt_path):
            print(f" ⚠️ Checkpoint file not found: {ckpt_path}. Skipping.")

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
        # --- 2. Loop through each Nepali Text Encoder ---
        self.nep_clip_encoder.to(device).eval()
        txt_embs = []

        tokenized_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        num_texts = tokenized_inputs['input_ids'].shape[0]
        with torch.no_grad():
            for i in tqdm(range(0, num_texts, batch_size), desc=f"Encoding Text for {self.name}"):
                batch_text = {k: v[i: i + batch_size].to(device) for k, v in tokenized_inputs.items()}
                txt_f = self.nep_clip_encoder(batch_text)
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
                txt_embs.append(txt_f.cpu())

        return torch.cat(txt_embs, dim=0).numpy()  # [N_texts, D]

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
        return torch.cat(img_embs, dim=0).numpy()  # [N_images, D]