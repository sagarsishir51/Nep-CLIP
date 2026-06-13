from PIL import Image
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

class EvaluationDataset(Dataset):
    def __init__(self, image_paths, captions,eng_captions, image_processor):
        print(f"Pre-processing {len(image_paths)} images and {len(captions)} captions...")

        # 1. Pre-process Images with a Loading Bar
        processed_images = []

        for path in tqdm(image_paths, desc="Processing Images", unit="img"):
            img = Image.open(path).convert("RGB")
            encoding = image_processor(images=img, return_tensors="pt")
            processed_images.append(encoding.pixel_values.squeeze(0))

        self.pixel_values = torch.stack(processed_images, dim=0)
        self.text = captions
        self.eng_text = eng_captions
        print("Dataset preparation complete!")

    def __len__(self):
        # Return length of captions (usually the primary index for datasets)
        return len(self.text_inputs['input_ids'])

    def get_all_data(self):
        """Helper to get everything for run_report"""
        return self.pixel_values, self.text,self.eng_text