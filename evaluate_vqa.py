from collections import defaultdict

import pandas as pd
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import numpy as np
from load_model import get_all_models_with_loaded_instance
from dataset_utils import get_device
from metrics import log_vqa_report
from vqa_dataset_utils import preprocess_vqa_datasets
from dotenv import load_dotenv

load_dotenv()
LOCAL_MODEL_PATH = "Nep_CLIP_Models"
TEACHER_ID = "openai/clip-vit-large-patch14-336"
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

def run_full_report_old(model_configs,dataset, device, batch_size):
    # 1. Get Image Embeddings once
    results = {}

    # 2. Loop through each Nepali Text Encoder
    for name, model in model_configs.items():
        results = []
        correct_count = 0
        total_count = 0
        for data in dataset:
            image_values = data["image_values"]
            text_inputs = data["text_inputs"]
            eng_text_inputs = data["eng_text_inputs"]
            correct_idx = data["correct_idx"]

            img_emb = model.get_image_features(image_values=image_values, batch_size=batch_size, device=device)
            if name == "CLIP-English":
                txt_emb = model.get_text_features(texts=eng_text_inputs, batch_size=batch_size, device = device)
            else:
                txt_emb = model.get_text_features(texts=text_inputs, batch_size=batch_size, device=device)
            # 3. Compute Similarity
            similarity = img_emb @ txt_emb.T
            p_idx, conf, all_scores = predict_best_option(
                similarity=similarity
            )

            # Store results
            res_item = data.copy()
            res_item['predicted_idx'] = p_idx
            res_item['confidence'] = conf
            res_item['all_scores'] = all_scores
            res_item['is_correct'] = (p_idx == correct_idx)
            results.append(res_item)

            # Update counters
            total_count += 1
            if res_item['is_correct']:
                correct_count += 1

        # --- Calculate and Print Accuracy ---
        if total_count > 0:
            overall_accuracy = (correct_count / total_count) * 100
            print("\n" + "=" * 30)
            print(f"FINAL EVALUATION SUMMARY")
            print("-" * 30)
            print(f"Total Processed: {total_count}")
            print(f"Total Correct:   {correct_count}")
            print(f"Overall Accuracy: {overall_accuracy:.2f}%")
            print("=" * 30)
        else:
            print("No items were processed.")

        # Memory management
        torch.cuda.empty_cache()


def run_full_report(model_configs,dataset, device, batch_size):
    # 1. Get Image Embeddings once
    results = {}

    # 2. Loop through each Nepali Text Encoder
    for name, model in model_configs.items():
        image_values = dataset["image_values"]
        text_inputs = dataset["text_inputs"]
        eng_text_inputs = dataset["eng_text_inputs"]
        labels = dataset["labels"]
        subsets = dataset["subsets"]
        categories = dataset["categories"]

        img_emb = model.get_image_features(image_values=image_values, batch_size=batch_size, device=device)
        if name == "CLIP-English":
            txt_emb = model.get_text_features(texts=eng_text_inputs, batch_size=batch_size, device = device)
        else:
            txt_emb = model.get_text_features(texts=text_inputs, batch_size=batch_size, device=device)
        # 3. Compute Similarity
        similarity = torch.mm(img_emb, txt_emb.t())
        num_images, total_texts = similarity.shape
        pred_indices, confidences, all_option_scores = process_dynamic_similarity(similarity_matrix=similarity)

        subset_counts = defaultdict(int)
        subset_correct = defaultdict(int)
        category_counts = defaultdict(int)
        category_correct = defaultdict(int)

        # Count correct predictions per group
        for pred, label, subset, category in zip(pred_indices, labels, subsets, categories):
            # Subset tracking
            subset_counts[subset] += 1
            if pred == label:
                subset_correct[subset] += 1

            # Category tracking
            category_counts[category] += 1
            if pred == label:
                category_correct[category] += 1

        # Calculate individual accuracies
        subset_accuracies = {}
        for subset in subset_counts:
            subset_accuracies[subset] = (subset_correct[subset] / subset_counts[subset]) * 100

        category_accuracies = {}
        for cat in category_counts:
            category_accuracies[cat] = (category_correct[cat] / category_counts[cat]) * 100

        # Calculate Macro Accuracies
        macro_accuracy_subsets = np.mean(list(subset_accuracies.values()))
        macro_accuracy_categories = np.mean(list(category_accuracies.values()))

        # Micro/Overall accuracy
        overall_correct = sum(subset_correct.values())
        overall_accuracy = (overall_correct / num_images) * 100

        results[name] = {
            "num_correct": overall_correct,
            "total_records": num_images,
            "overall_accuracy_percentage": overall_accuracy,

            # Subset Data
            "macro_accuracy_subsets": macro_accuracy_subsets,
            "subset_accuracies": subset_accuracies,

            # Category Data
            "macro_accuracy_categories": macro_accuracy_categories,
            "category_accuracies": category_accuracies,

            "predictions": pred_indices,
            "confidences": confidences,
            "all_scores": all_option_scores
        }
        # Memory management
        torch.cuda.empty_cache()
    return {
        "results": results,
    }

if __name__ == "__main__":
    DEVICE = get_device()
    clip_processor = CLIPProcessor.from_pretrained(TEACHER_ID)
    print("============================================= VQA Process for CLIP Model started =============================================")
    model_configs = get_all_models_with_loaded_instance(model_folder=LOCAL_MODEL_PATH, device=DEVICE)
    processed_datasets = preprocess_vqa_datasets(clip_processor=clip_processor)
    for dataset_name, data in processed_datasets.items():
        print(
            f"=================================================================== Processing {dataset_name} ===================================================================")
        result = run_full_report(
            model_configs=model_configs,
            dataset=data,
            device=DEVICE,
            batch_size=BATCH_SIZE,
        )
        log_vqa_report(result["results"])
