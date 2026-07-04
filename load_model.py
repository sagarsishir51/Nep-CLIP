from transformers import AutoTokenizer, CLIPProcessor,CLIPModel
import torch
import transformers
from transformers import AutoConfig
from huggingface_hub import hf_hub_download

from model.CLIP_Model import CLIP_Model
from model.MCLIP_Model import MCLIP_Model
from model.NLLB_Model import NLLB_Model
from model.NepClip_Model import NepClip_Model

TEACHER_ID = "openai/clip-vit-base-patch32"

DISTILLED_MODELS_CONFIG = {
    # "BERT": {
    #     "name":"BERT",
    #     "id": "Shushant/nepaliBERT",
    #     "teacher_id": "openai/clip-vit-base-patch32",
    #     "file_name": "Shushant-nepaliBERT distillation_epoch_4_step_150000.pt"
    # },
    # "RoBERTa":
    #     {
    #         "name": "RoBERTa",
    #         "id": "IRIISNEPAL/RoBERTa_Nepali_110M",
    #          "teacher_id": "openai/clip-vit-base-patch32",
    #         "file_name": "RoBERTa_Nepali_110M distillation_epoch_4_step_150000.pt"
    #     },
    # "NLLB-CLIP":
    #     {
    #         "name": "NLLB",
    #         "id": "facebook/nllb-200-distilled-600M",
    #         "teacher_id":"openai/clip-vit-base-patch32",
    #         "repo_id":"visheratin/nllb-clip-base",
    #     },
    # "MCLIP":{
    #     "name": "MCLIP",
    #     "id": "M-CLIP/XLM-Roberta-Large-Vit-B-32",
    #     "teacher_id":"openai/clip-vit-base-patch32",
    # },
    "CLIP-English":{
        "name": "CLIP-English",
        "teacher_id":"openai/clip-vit-large-patch14-336",
        },
    # "CLIP (Nepali input)": {
    #     "name": "CLIP (Nepali input)",
    #     "teacher_id": "openai/clip-vit-base-patch32",
    # }
}

def get_all_models_with_loaded_instance(model_folder, device):
    models_dict = {}
    required_version = "4.38.2"
    current_version = transformers.__version__

    # Check if we are in the specific 4.38.2 environment
    is_target_env = (current_version == required_version)

    for student_key, config in DISTILLED_MODELS_CONFIG.items():
        print(f"\n--- Initializing model for: {student_key} ---")

        if is_target_env:
            # ENVIRONMENT A (4.38.2): Load ONLY NLLB and MCLIP. Neglect all others.
            if student_key == "NLLB-CLIP":
                print(f"Correct version ({current_version}) confirmed. Loading NLLB...")
                models_dict[student_key] = NLLB_Model(config=config)
            elif student_key == "MCLIP":
                print(f"Correct version ({current_version}) confirmed. Loading MCLIP...")
                models_dict[student_key] = MCLIP_Model(config=config)
            else:
                print(f"Version is 4.38.2. Neglecting model: {student_key}")

        else:
            # ENVIRONMENT B (Not 4.38.2): Neglect NLLB/MCLIP. Load everything else.
            if student_key in ["NLLB-CLIP", "MCLIP"]:
                print(f"Skipping {student_key}: Requires v{required_version} (Found v{current_version}).")
            elif student_key in ["CLIP-English", "CLIP (Nepali input)"]:
                print(f"Loading {student_key} standard instance...")
                models_dict[student_key] = CLIP_Model(config=config)
            else:
                print(f"Loading {student_key} fallback instance...")
                models_dict[student_key] = NepClip_Model(config=config, model_path=model_folder, device=device)

    return models_dict