from PIL import Image
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import ast

class VQAEvaluationDataset(Dataset):
    def __init__(self, dataset_translated_map, dataset, image_processor):
        """
        Memory-safe Lazy Loading Dataset.
        """
        self.texts = []
        self.eng_texts = []
        self.location_aware_eng_texts = []
        self.labels = []
        self.locations = []
        self.categories = []
        processed_images = []

        for item in tqdm(dataset, desc="Processing Images", unit="img"):
            img = item.get("image")
            id = item.get("ID")
            translated_dataset = dataset_translated_map[id]

            if isinstance(img, Image.Image):
                img = img
            elif not isinstance(img, torch.Tensor):
                img = Image.open(img).convert("RGB")
            else:
                img = img
            encoding = image_processor(images=img, return_tensors="pt")
            processed_images.append(encoding.pixel_values.squeeze(0))
            self.labels.append(item.get("Label"))
            nepali_question = translated_dataset.get("nepali_question")
            translated_options = translated_dataset.get("nepali_options")
            self.texts.extend([f"{nepali_question} {opt}" for opt in translated_options])

            english_question = item.get("Translated Question")
            english_options = item.get("Translated Options")
            subset = item.get("Subset")
            subset_tuple = ast.literal_eval(subset)
            self.locations.append(subset_tuple[1])
            self.categories.append(item.get("Category"))
            # self.eng_texts.extend([f"{english_question} {opt}" for opt in english_options])
            self.eng_texts.extend([f"Question: {english_question} Options: {opt} Short Answer:" for opt in english_options])
            self.location_aware_eng_texts.extend([f"Location: {subset_tuple[1]}. Question: {english_question} Options: {opt} Short Answer:" for opt in english_options])

        self.pixel_values = torch.stack(processed_images, dim=0)

        print(f"Dataset initialized safely with {len(self.texts)} samples.")

    def __len__(self):
        return len(self.texts)

    def get_all_data(self):
        """Helper to get everything for run_report"""
        return self.pixel_values,self.texts,self.eng_texts,self.location_aware_eng_texts,self.labels,self.locations,self.categories