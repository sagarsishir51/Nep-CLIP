from PIL import Image
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

class EvaluationDataset(Dataset):
    def __init__(self, image_paths, test_texts,test_eng_texts, image_processor):
        print(f"Pre-processing {len(image_paths)} images and {len(test_texts)} captions...")

        # 1. Pre-process Images with a Loading Bar
        processed_images = []

        for path in tqdm(image_paths, desc="Processing Images", unit="img"):
            if not isinstance(path, torch.Tensor):
                img = Image.open(path).convert("RGB")
            else:
                img = path
            encoding = image_processor(images=img, return_tensors="pt")
            processed_images.append(encoding.pixel_values.squeeze(0))

        self.pixel_values = torch.stack(processed_images, dim=0)
        self.texts = test_texts
        self.eng_texts = test_eng_texts
        print("Dataset preparation complete!")

    def __len__(self):
        # Return length of captions (usually the primary index for datasets)
        return len(self.text_inputs['input_ids'])

    def get_all_data(self):
        """Helper to get everything for run_report"""
        return self.pixel_values, self.texts,self.eng_texts