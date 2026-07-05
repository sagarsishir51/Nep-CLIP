import pandas as pd
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

from load_model import get_all_models_with_loaded_instance, TEACHER_ID
from metrics import compute_metrics, log_report
from dataset_utils import get_device, preprocess_all_datasets
from dotenv import load_dotenv

load_dotenv()
LOCAL_MODEL_PATH = "Nep_CLIP_Models"
MAX_LENGTH = 77
BATCH_SIZE = 256

def run_full_report(model_configs,image_values,text_inputs,eng_text_inputs, device, batch_size,captions_per_image):
    # 1. Get Image Embeddings once
    results = {}

    # 2. Loop through each Nepali Text Encoder
    for name, model in model_configs.items():
        img_emb = model.get_image_features(image_values=image_values, batch_size=batch_size, device=device)
        if name == "CLIP-English":
            txt_emb = model.get_text_features(texts=eng_text_inputs, batch_size=batch_size, device = device)
        else:
            txt_emb = model.get_text_features(texts=text_inputs, batch_size=batch_size, device=device)
        # 3. Compute Similarity
        similarity = img_emb @ txt_emb.T
        # 4. Store metrics
        results[name] = {
            "txt_emb": txt_emb,
            "I2T": compute_metrics(similarity, query_type='i2t',captions_per_image=captions_per_image),
            "T2I": compute_metrics(similarity, query_type='t2i',captions_per_image=captions_per_image)
        }

        # Memory management
        torch.cuda.empty_cache()
    return {
        "results": results,
    }


if __name__ == "__main__":
    DEVICE = get_device()
    clip_processor = CLIPProcessor.from_pretrained(TEACHER_ID)
    print("============================================= Image Retrieval Process for CLIP Model started =============================================")
    model_configs = get_all_models_with_loaded_instance(model_folder=LOCAL_MODEL_PATH, device=DEVICE)
    processed_datasets = preprocess_all_datasets(clip_processor=clip_processor)
    for dataset_name, data in processed_datasets.items():
        print(f"=================================================================== Processing {dataset_name} ===================================================================")
        image_values = data["image_values"]
        text_inputs = data["text_inputs"]
        eng_text_inputs = data["eng_text_inputs"]
        captions_per_image = data["captions_per_image"]
        complete_reports = run_full_report(
            model_configs=model_configs,
            image_values=image_values,
            text_inputs=text_inputs,
            eng_text_inputs=eng_text_inputs,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            captions_per_image=captions_per_image
        )

        log_report(complete_reports["results"])
