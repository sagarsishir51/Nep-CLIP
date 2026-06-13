import pandas as pd
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os
from metrics import compute_metrics, log_report
from model.NepClipEncoder import NepClipEncoder
from dataset_utils import get_device, preprocess_all_datasets

LOCAL_MODEL_PATH = "Nep_CLIP_Models"
TEACHER_ID = "openai/clip-vit-base-patch32"
MAX_LENGTH = 77
BATCH_SIZE = 256

DISTILLED_MODELS_CONFIG = {
    "BERT": {
        "id": "Shushant/nepaliBERT",
        "file_name": "Shushant-nepaliBERT distillation_epoch_4_step_150000.pt"
    },
    # "RoBERTa":
    #     {
    #         "id": "IRIISNEPAL/RoBERTa_Nepali_110M",
    #         "file_name": "RoBERTa_Nepali_110M distillation_epoch_4_step_150000.pt"
    #     }
}


def get_image_features(clip_model, image_values, batch_size=256, device="cuda"):
    img_embs = []
    clip_model.to(device).eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(image_values), batch_size), desc="Encoding Images"):
            batch_pixels = image_values[i: i + batch_size].to(device)
            img_f = clip_model.get_image_features(pixel_values=batch_pixels)
            # Handle potential pooler_output or raw output depending on CLIP version
            if hasattr(img_f, 'pooler_output'):
                img_f = img_f.pooler_output

            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            img_embs.append(img_f.cpu())
    return torch.cat(img_embs, dim=0).numpy()  # [N_images, D]


def get_text_features(encoder, encoder_name, text_inputs, batch_size, device):
    # --- 2. Loop through each Nepali Text Encoder ---
    encoder.to(device).eval()
    txt_embs = []

    num_texts = text_inputs['input_ids'].shape[0]
    with torch.no_grad():
        for i in tqdm(range(0, num_texts, batch_size), desc=f"Encoding Text ({encoder_name})"):
            batch_text = {k: v[i: i + batch_size].to(device) for k, v in text_inputs.items()}
            txt_f = encoder(batch_text)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            txt_embs.append(txt_f.cpu())

    return torch.cat(txt_embs, dim=0).numpy()  # [N_texts, D]


def run_full_report(text_encoders_dict, img_emb, text_inputs, device, batch_size):
    # 1. Get Image Embeddings once
    results = {}

    # 2. Loop through each Nepali Text Encoder
    for name, encoder in text_encoders_dict.items():
        nep_clip_encoder = encoder["nep_clip_encoder"]
        tokenizer = encoder["tokenizer"]

        print(f"\nEvaluating: {name}")
        print("Pre-tokenizing text inputs")
        tokenized_inputs = tokenizer(
            text_inputs,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        txt_emb = get_text_features(nep_clip_encoder, name, tokenized_inputs, batch_size, device)

        # 3. Compute Similarity
        similarity = img_emb @ txt_emb.T

        # 4. Store metrics
        results[name] = {
            "txt_emb": txt_emb,
            "I2T": compute_metrics(similarity, query_type='i2t'),
            "T2I": compute_metrics(similarity, query_type='t2i')
        }

        # Memory management
        nep_clip_encoder.cpu()
        torch.cuda.empty_cache()
    return {
        "results": results,
        "img_emb": img_emb
    }


def get_all_models_with_loaded_instance(model_folder, device):
    encoders_dict = {}

    # Loop through each student in the configuration
    for student_key, config in DISTILLED_MODELS_CONFIG.items():
        print(f"\n--- Initializing model for: {student_key} ---")

        # Extract the model ID and its specific checkpoint filename
        student_id = config["id"]
        ckpt_name = config["file_name"]

        print(f"Loading weights from: {ckpt_name}")
        ckpt_path = os.path.join(model_folder, ckpt_name)

        # Check if the file actually exists before processing
        if not os.path.exists(ckpt_path):
            print(f" ⚠️ Checkpoint file not found: {ckpt_path}. Skipping.")
            continue

        # 2. Initialize a fresh instance of the encoder using the HuggingFace ID
        encoder = NepClipEncoder(student_id)

        # 3. Load the checkpoint to the target device safely
        ckpt = torch.load(ckpt_path, map_location=device)

        # 4. Load state dicts into the sub-modules
        if isinstance(ckpt, dict) and "student_state_dict" in ckpt and "projection_state_dict" in ckpt:
            # Custom dictionary tracking structure
            encoder.student.load_state_dict(ckpt["student_state_dict"])
            encoder.projection.load_state_dict(ckpt["projection_state_dict"])
        else:
            # Fallback: If the checkpoint itself IS the raw state dict
            try:
                encoder.load_state_dict(ckpt)
            except Exception as e:
                print(f" ⚠️ Could not automatically parse weights for {ckpt_name}. Error: {e}")
                continue

        # 5. Send the entire encoder model to the selected device (MPS/CUDA/CPU)
        encoder.to(device)
        best_kw = {'do_lower_case': False, 'strip_accents': False}
        tokenizer = AutoTokenizer.from_pretrained(student_id,**best_kw)

        # Set to evaluation mode
        encoder.eval()

        # Store inside the structure under the configuration key
        encoders_dict[student_key] = {
            "nep_clip_encoder":encoder,
            "tokenizer":tokenizer,
        }

    return encoders_dict


if __name__ == "__main__":
    DEVICE = get_device()
    clip_processor = CLIPProcessor.from_pretrained(TEACHER_ID)
    clip_model = CLIPModel.from_pretrained(TEACHER_ID)
    processed_datasets = preprocess_all_datasets(clip_processor=clip_processor)
    for dataset_name, data in processed_datasets.items():
        image_values = data["image_values"]
        text_inputs = data["text_inputs"]
        img_emb = get_image_features(clip_model=clip_model, image_values=image_values, batch_size=BATCH_SIZE, device=DEVICE)
        text_encoders_dict = get_all_models_with_loaded_instance(model_folder=LOCAL_MODEL_PATH, device=DEVICE)
        complete_reports = run_full_report(
            text_encoders_dict=text_encoders_dict,
            img_emb=img_emb,
            text_inputs=text_inputs,
            device=DEVICE,
            batch_size=BATCH_SIZE
        )
        log_report(complete_reports["results"])
