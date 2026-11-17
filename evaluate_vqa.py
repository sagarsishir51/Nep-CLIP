from collections import defaultdict

import pandas as pd
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np
from load_model import get_all_models_with_loaded_instance, TEACHER_ID
from dataset_utils import get_device
from metrics import log_vqa_report
from vqa_dataset_utils import preprocess_vqa_datasets
from dotenv import load_dotenv
import argparse

load_dotenv()
LOCAL_MODEL_PATH = "Nep_CLIP_Models"
BATCH_SIZE = 256

def predict_best_option(similarity):
    with torch.no_grad():

        # 4. Calculate Scores
        logits = similarity.squeeze()
        probs = (logits * 100).softmax(dim=-1)

        # Convert all scores to a list of floats
        option_scores = probs.cpu().tolist()

        pred_idx = probs.argmax().item()
        confidence = option_scores[pred_idx]

    return pred_idx, confidence, option_scores


def process_dynamic_similarity(similarity_matrix):
    """
    Args:
        similarity_matrix (Tensor): Shape [N, N * K]
                                    where N = num_images, (N * K) = total_texts
    """
    # 1. Dynamically infer N (num_images) and total_texts
    num_images, total_texts = similarity_matrix.shape

    # 2. Calculate K (text options per image) dynamically
    num_options = total_texts // num_images

    # 3. Reshape to [N, N, K] to isolate groups of options
    similarity_grouped = similarity_matrix.view(num_images, num_images, num_options)

    # 4. Extract the matching diagonal blocks -> Shape: [N, num_options]
    # torch.diagonal returns [num_options, N], so we permute back to [N, num_options]
    logits = torch.diagonal(similarity_grouped, dim1=0, dim2=1).permute(1, 0)

    # 5. Calculate probabilities and predictions dynamically
    with torch.no_grad():
        probs = (logits * 100).softmax(dim=-1)  # Shape: [N, num_options]

        pred_indices = probs.argmax(dim=-1).cpu().tolist()  # List of N ints
        all_option_scores = probs.cpu().tolist()  # Nested list of shape [N, num_options]

        # Extract confidence for each image's chosen option
        confidences = [scores[idx] for idx, scores in zip(pred_indices, all_option_scores)]

    return pred_indices, confidences, all_option_scores

def run_full_report(model_configs, dataset, device, batch_size):
    # 1. Get Image Embeddings once
    results = {}

    # 2. Loop through each Nepali Text Encoder
    for name, model in model_configs.items():
        image_values = dataset["image_values"]
        text_inputs = dataset["text_inputs"]
        eng_text_inputs = dataset["eng_text_inputs"]
        location_aware_eng_text_inputs = dataset.get("location_aware_eng_text_inputs") or []
        labels = dataset["labels"]
        locations = dataset["locations"]
        categories = dataset["categories"]
        print("image_values",len(image_values))
        print("eng_text_inputs",len(eng_text_inputs))
        img_emb = model.get_image_features(image_values=image_values, batch_size=batch_size, device=device)

        if name == "CLIP-English":
            txt_emb = model.get_text_features(texts=eng_text_inputs, batch_size=batch_size, device=device)
            location_specific_txt_emb = model.get_text_features(texts=location_aware_eng_text_inputs,
                                                                batch_size=batch_size, device=device)
        else:
            txt_emb = model.get_text_features(texts=text_inputs, batch_size=batch_size, device=device)
            # Assuming non-English models use standard text or handles location embedding similarly if provided
            location_specific_txt_emb = model.get_text_features(texts=text_inputs, batch_size=batch_size, device=device)

        # 3. Compute Similarity Matrices
        similarity = torch.mm(img_emb, txt_emb.t())
        location_specific_similarity = torch.mm(img_emb, location_specific_txt_emb.t())
        num_images, total_texts = similarity.shape

        # Process predictions for both standard and location-specific similarities
        pred_indices, confidences, all_option_scores = process_dynamic_similarity(similarity_matrix=similarity)
        loc_pred_indices, _, _ = process_dynamic_similarity(similarity_matrix=location_specific_similarity)

        location_counts = defaultdict(int)
        location_correct = defaultdict(int)
        category_counts = defaultdict(int)
        category_correct = defaultdict(int)

        # Count correct predictions per group
        for pred, loc_pred, label, location, category in zip(pred_indices, loc_pred_indices, labels, locations,
                                                             categories):
            # Location tracking (Uses location_specific_similarity predictions)
            location_counts[location] += 1
            if loc_pred == label:
                location_correct[location] += 1

            # Category tracking (Uses standard similarity predictions)
            category_counts[category] += 1
            if pred == label:
                category_correct[category] += 1

        # Calculate individual accuracies
        location_accuracies = {}
        for location in location_counts:
            location_accuracies[location] = (location_correct[location] / location_counts[location]) * 100

        category_accuracies = {}
        for cat in category_counts:
            category_accuracies[cat] = (category_correct[cat] / category_counts[cat]) * 100

        # Calculate Macro Accuracies
        macro_accuracy_locations = np.mean(list(location_accuracies.values())) if location_accuracies else 0.0
        macro_accuracy_categories = np.mean(list(category_accuracies.values())) if category_accuracies else 0.0

        # Micro/Overall accuracy (Still based on standard text predictions)
        overall_correct = sum(1 for p, l in zip(pred_indices, labels) if p == l)
        overall_accuracy = (overall_correct / num_images) * 100

        results[name] = {
            "num_correct": overall_correct,
            "total_records": num_images,
            "overall_accuracy_percentage": overall_accuracy,

            # Locations Data (Now location_specific driven)
            "macro_accuracy_locations": macro_accuracy_locations,
            "location_accuracies": location_accuracies,

            # Category Data
            "macro_accuracy_categories": macro_accuracy_categories,
            "category_accuracies": category_accuracies,

            "predictions": pred_indices,
            "location_predictions": loc_pred_indices,
            "confidences": confidences,
            "all_scores": all_option_scores
        }

        # Memory management
        torch.cuda.empty_cache()

    return {
        "results": results,
    }


if __name__ == "__main__":
    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Run VQA Process for CLIP Models.")

    # 2. Add the batch_size argument (with a default value just in case)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for running the full report (default: 32)"
    )

    # 3. Parse the arguments
    args = parser.parse_args()

    # Use args.batch_size in your script
    DEVICE = get_device()
    clip_processor = CLIPProcessor.from_pretrained(TEACHER_ID)
    batch_size = args.batch_size
    print(
        f"============================================= VQA Process for CLIP Model started with batch size {batch_size}. =============================================")
    model_configs = get_all_models_with_loaded_instance(model_folder=LOCAL_MODEL_PATH, device=DEVICE)
    processed_datasets = preprocess_vqa_datasets(clip_processor=clip_processor)

    for dataset_name, data in processed_datasets.items():
        print(
            f"=================================================================== Processing {dataset_name} ===================================================================")
        result = run_full_report(
            model_configs=model_configs,
            dataset=data,
            device=DEVICE,
            batch_size=batch_size,
        )
        log_vqa_report(result["results"])
