import json
import os

import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torchvision
from datasets import load_dataset

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

DATASET_LIST = ["CVQA"]


def preprocess_vqa_datasets(clip_processor):
    processed_datasets = {}
    for dataset_name in DATASET_LIST:
        if dataset_name == "CVQA":
            try:
                cvqa_data = load_cvqa_dataset()
                cvqa_translated_data_map = load_cvqa_translated_dataset()
                # cvqa_data = cvqa_data.select(range(100))
                dataset = VQAEvaluationDataset(dataset=cvqa_data, dataset_translated_map = cvqa_translated_data_map, image_processor=clip_processor)
                image_values, text_inputs, eng_text_inputs,labels,subsets,categories = dataset.get_all_data()
                processed_datasets[dataset_name] = {
                    "image_values": image_values,
                    "text_inputs": text_inputs,
                    "eng_text_inputs": eng_text_inputs,
                    "labels": labels,
                    "subsets": subsets,
                    "categories": categories
                }
                # processed_datasets[dataset_name] = dataset

            except Exception as e:
                print(e)
                print("CVQA dataset not found. First download dataset using script download_dataset.py")
                continue
        else:
            print(f"Dataset {dataset_name} not found.")
    return processed_datasets