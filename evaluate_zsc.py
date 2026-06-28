import pandas as pd
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

from load_model import get_all_models_with_loaded_instance
from metrics import compute_zsc_metrics, log_zsc_report
from dataset_utils import get_device
from zsc_dataset_utils import preprocess_zsc_datasets
from dotenv import load_dotenv

load_dotenv()
LOCAL_MODEL_PATH = "Nep_CLIP_Models"
TEACHER_ID = "openai/clip-vit-base-patch32"
BATCH_SIZE = 256


def run_full_report(model_configs,image_values,text_inputs,eng_text_inputs,gt_labels,gt_labels_en, device, batch_size,top_k=5):
    results = {}

    for name, model in model_configs.items():
        img_emb = model.get_image_features(image_values=image_values, batch_size=batch_size, device=device)
        if name == "CLIP-English":
            txt_emb = model.get_text_features(texts=eng_text_inputs, batch_size=batch_size, device = device)
            results[name] = compute_zsc_metrics(
                img_f=img_emb,
                txt_f=txt_emb,
                gt_labels=gt_labels_en if len(gt_labels_en) > 0 else gt_labels,
                top_k=top_k,
                device=device,
            )
        else:
            txt_emb = model.get_text_features(texts=text_inputs, batch_size=batch_size, device=device)
            results[name] = compute_zsc_metrics(
                img_f=img_emb,
                txt_f=txt_emb,
                gt_labels=gt_labels,
                top_k=top_k,
                device=device,
            )

        # Memory management
        torch.cuda.empty_cache()
    return {
        "results": results,
    }

if __name__ == "__main__":
    DEVICE = get_device()
    clip_processor = CLIPProcessor.from_pretrained(TEACHER_ID)
    print("============================================= Zero Shot Classification Process for CLIP Model started =============================================")
    model_configs = get_all_models_with_loaded_instance(model_folder=LOCAL_MODEL_PATH, device=DEVICE)
    processed_datasets = preprocess_zsc_datasets(clip_processor=clip_processor)
    for dataset_name, data in processed_datasets.items():
        print(f"=================================================================== Processing {dataset_name} ===================================================================")
        image_values = data["image_values"]
        text_inputs = data["text_inputs"]
        eng_text_inputs = data["eng_text_inputs"]
        gt_labels = data["gt_labels"]
        gt_labels_en = data["gt_labels_en"]
        complete_reports = run_full_report(
            model_configs=model_configs,
            image_values=image_values,
            text_inputs=text_inputs,
            eng_text_inputs=eng_text_inputs,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            gt_labels=gt_labels,
            gt_labels_en=gt_labels_en,
        )
        log_zsc_report(complete_reports["results"])
