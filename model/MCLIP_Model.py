from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, CLIPProcessor,CLIPModel
import torch
from tqdm.auto import tqdm

class MCLIP_Model:
    def __init__(self,config):
        student_id = config["id"]
        teacher_id = config["teacher_id"]
        from multilingual_clip import pt_multilingual_clip
        self.model = CLIPModel.from_pretrained(teacher_id)
        self.processor = CLIPProcessor.from_pretrained(teacher_id)
        self.tokenizer = AutoTokenizer.from_pretrained(student_id)
        self.mclip = pt_multilingual_clip.MultilingualCLIP.from_pretrained(student_id)

    def get_text_features(self, texts, batch_size=256,device="cpu"):
        txt_embs = []
        num_texts = len(texts)
        with torch.no_grad():
            # tqdm for progress tracking
            for i in tqdm(range(0, num_texts, batch_size), desc="Encoding Text"):
                batch_text = texts[i: i + batch_size]

                # The .forward() method takes the raw list of strings,
                # tokenizes them, and moves them to the device automatically.
                txt_f = self.mclip.forward(batch_text, self.tokenizer)

                # Normalize for cosine similarity
                txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

                # Move to CPU to save GPU memory
                txt_embs.append(txt_f.cpu())

        return torch.cat(txt_embs, dim=0).numpy()  # [N_texts, D]


    def get_image_features(self, image_values, batch_size=256, device="cuda"):
        img_embs = []
        self.model.to(device).eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(image_values), batch_size), desc="Encoding Images"):
                batch_pixels = image_values[i: i + batch_size].to(device)
                img_f = self.model.get_image_features(pixel_values=batch_pixels)
                if hasattr(img_f, 'pooler_output'):
                    img_f = img_f.pooler_output

                img_f = img_f / img_f.norm(dim=-1, keepdim=True)
                img_embs.append(img_f.cpu())
        return torch.cat(img_embs, dim=0).numpy()  # [N_images, D]