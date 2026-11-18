import json
import os
import time

import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torchvision
from datasets import load_dataset

from ALMEvaluationDataset import ALMEvaluationDataset
from EvaluationDataset import EvaluationDataset
from VQAEvaluationDataset import VQAEvaluationDataset

CVQA_JSON_FILE_NAME="Nep_CLIP_Datasets/cvqa_translated.json"


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def load_split(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f]


def get_test_images(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    test_images_ids = []

    for img in data['images']:
        if img['split'] == 'test':
            test_images_ids.append(img['filename'])
    return test_images_ids


def load_cvqa_dataset():
    """
    Loads the CVQA dataset from the local target folder.
    If the local cache isn't found, it safely falls back to loading it properly.
    """
    print("\n[Loading CVQA Dataset]")
    target_dir = "dataset/cvqa"

    if not os.path.exists(target_dir) or len(os.listdir(target_dir)) == 0:
        print(f"  ❌ Error: Local dataset not found in '{target_dir}'. Please run the download function first.")
        return None

    try:
        # Point Hugging Face back to the local cache directory we saved it to
        print(f"  Loading CVQA 'test' split from local directory: '{target_dir}'...")
        REPO = "afaji/cvqa"
        dataset = load_dataset(REPO, split="test", cache_dir=target_dir)

        print(f"  ✅ Success! Loaded {len(dataset)} samples.")
        print("  Sample features available:", list(dataset.features.keys()))

        return dataset

    except Exception as e:
        print(f"  ❌ Error loading CVQA dataset: {e}")
        return None

def load_cvqa_translated_dataset():
    with open(CVQA_JSON_FILE_NAME, "r", encoding="utf-8") as f:
        translated_list = json.load(f)

    return {
        item["ID"]: {
            "nepali_question": item.get("Translated Question Nepali"),
            "nepali_options": item.get("Translated Options Nepali")
        }
        for item in translated_list if "ID" in item
    }


def load_alm_bench_dataset():
    """
    Loads the MBZUAI/ALM-Bench dataset from the local target folder.
    If the local cache isn't found, it safely falls back to loading it properly.
    """
    print("\n[Loading ALM-Bench Dataset]")
    target_dir = "dataset/alm_bench"

    if not os.path.exists(target_dir) or len(os.listdir(target_dir)) == 0:
        print(f"  ❌ Error: Local dataset not found in '{target_dir}'. Please run the download function first.")
        return None

    try:
        # Point Hugging Face back to the local cache directory we saved it to
        print(f"  Loading ALM-Bench dataset from local directory: '{target_dir}'...")
        REPO = "MBZUAI/ALM-Bench"
        dataset = load_dataset(REPO, cache_dir=target_dir)

        print(f"  ✅ Success! Loaded dataset successfully.")
        print("  Available splits:", list(dataset.keys()))

        # Pull a sample split to showcase available features
        first_split = list(dataset.keys())[0]
        print(f"  Sample features available ({first_split}):", list(dataset[first_split].features.keys()))

        return dataset

    except Exception as e:
        print(f"  ❌ Error loading dataset: {e}")
        return None


def get_samples_by_type(dataset_dict, question_type="Multiple Choice Questions", split="test"):
    """
    Instantly filters the Hugging Face dataset by Question_Type without loading
    heavy image objects into memory.
    """
    if split not in dataset_dict:
        print(f"❌ Split '{split}' not found in dataset.")
        return None

    dataset_split = dataset_dict[split]

    # 1. Inspect only the text column to bypass image decoding
    q_types_column = dataset_split["Question_Type"]

    # 2. Extract indices matching the target type
    matching_indices = [
        i for i, q_type in enumerate(q_types_column)
        if q_type == question_type
    ]

    # 3. Create a shallow pointer mapping to those indices
    filtered_dataset = dataset_split.select(matching_indices)
    print(f"   Found {len(filtered_dataset)} matches out of {len(dataset_split)} total rows.")

    return filtered_dataset

# DATASET_LIST = ["CVQA","ALM"]
DATASET_LIST = ["ALM"]


def preprocess_vqa_datasets(clip_processor):
    processed_datasets = {}
    for dataset_name in DATASET_LIST:
        if dataset_name == "CVQA":
            try:
                cvqa_data = load_cvqa_dataset()
                cvqa_translated_data_map = load_cvqa_translated_dataset()
                dataset = VQAEvaluationDataset(dataset=cvqa_data, dataset_translated_map = cvqa_translated_data_map, image_processor=clip_processor)
                image_values, text_inputs, eng_text_inputs,location_aware_eng_text_inputs,labels,locations,categories = dataset.get_all_data()
                processed_datasets[dataset_name] = {
                    "image_values": image_values,
                    "text_inputs": text_inputs,
                    "eng_text_inputs": eng_text_inputs,
                    "location_aware_eng_text_inputs": location_aware_eng_text_inputs,
                    "labels": labels,
                    "locations": locations,
                    "categories": categories
                }
                # processed_datasets[dataset_name] = dataset

            except Exception as e:
                print(e)
                print("CVQA dataset not found. First download dataset using script download_dataset.py")
                continue
        elif dataset_name == "ALM":
            try:
                aml_data = load_alm_bench_dataset()
                mcq_dataset = get_samples_by_type(dataset_dict=aml_data, question_type="Multiple Choice Questions")
                dataset = ALMEvaluationDataset(dataset=mcq_dataset, dataset_translated_map = None, image_processor=clip_processor)
                image_values, text_inputs, eng_text_inputs,labels,languages,categories = dataset.get_all_data()
                processed_datasets[dataset_name] = {
                    "image_values": image_values,
                    "text_inputs": text_inputs,
                    "eng_text_inputs": eng_text_inputs,
                    "location_aware_eng_text_inputs": eng_text_inputs,
                    "labels": labels,
                    "locations": languages,
                    "categories": categories
                }
            except Exception as e:
                print(e)
                print("ALM dataset not found. First download dataset using script download_dataset.py")
                continue
        else:
            print(f"Dataset {dataset_name} not found.")
    return processed_datasets