from transformers import AutoTokenizer, CLIPProcessor,CLIPModel
import torch
from transformers import AutoConfig
from huggingface_hub import hf_hub_download

from model.CLIP_Model import CLIP_Model
from model.MCLIP_Model import MCLIP_Model
from model.NLLB_Model import NLLB_Model
from model.NepClip_Model import NepClip_Model

TEACHER_ID = "openai/clip-vit-base-patch32"

DISTILLED_MODELS_CONFIG = {
    # "BERT": {
    #     "id": "Shushant/nepaliBERT",
    #     "teacher_id": "openai/clip-vit-base-patch32",
    #     "file_name": "Shushant-nepaliBERT distillation_epoch_4_step_150000.pt"
    # },
    # "RoBERTa":
    #     {
    #         "id": "IRIISNEPAL/RoBERTa_Nepali_110M",
    #          "teacher_id": "openai/clip-vit-base-patch32",
    #         "file_name": "RoBERTa_Nepali_110M distillation_epoch_4_step_150000.pt"
    #     },
    "NLLB":
        {
            "id": "facebook/nllb-200-distilled-600M",
            "teacher_id":"openai/clip-vit-base-patch32",
            "repo_id":"visheratin/nllb-clip-base",
        },
    # "MCLIP":{
    #     "id": "M-CLIP/XLM-Roberta-Large-Vit-B-32",
    #     "teacher_id":"openai/clip-vit-base-patch32",
    # },
    # "CLIP-English":{
    #         "teacher_id":"openai/clip-vit-base-patch32",
    #     },
    # "CLIP (Nepali input)": {
    #     "teacher_id": "openai/clip-vit-base-patch32",
    # }
}

def get_all_models_with_loaded_instance(model_folder, device):
    models_dict = {}

    # Loop through each student in the configuration
    for student_key, config in DISTILLED_MODELS_CONFIG.items():
        print(f"\n--- Initializing model for: {student_key} ---")
        if student_key == "NLLB":
            models_dict[student_key] = NLLB_Model(config=config)
        elif student_key == "MCLIP":
            models_dict[student_key] = MCLIP_Model(config=config)
        elif student_key == "CLIP-English" or student_key == "CLIP (Nepali input)":
            models_dict[student_key] = CLIP_Model(config=config)
        else:
            models_dict[student_key] = NepClip_Model(config=config, model_path=model_folder, device=device)

    return models_dict