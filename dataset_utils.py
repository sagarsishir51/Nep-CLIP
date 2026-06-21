import json
import os
import re

import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from EvaluationDataset import EvaluationDataset

FLICK_8K_CSV_FILE_NAME="Nep_CLIP_Datasets/flickr8k_nepali_output_google_translate.csv"
FLICK_30K_CSV_FILE_NAME="Nep_CLIP_Datasets/flickr30k_nepali_output_google_translate.csv"
XTD10_CSV_FILE_NAME="Nep_CLIP_Datasets/xtd10_google_translate.csv"
XM3600_CSV_FILE_NAME="Nep_CLIP_Datasets/xm3600_google_translate.csv"
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

def get_test_images_for_flickr(test_img_ids,image_path,flickr_dataset):
    flickr_dataset['image_id_update'] = flickr_dataset['image_id'].str.replace(r'(\.jpg).*', r'\1', regex=True)
    caption_dict = flickr_dataset.groupby('image_id_update')['translated_caption_np'].apply(list).to_dict()
    eng_caption_dict = flickr_dataset.groupby('image_id_update')['caption'].apply(list).to_dict()
    test_images = []
    test_captions = []
    test_eng_captions = []

    for img_id in test_img_ids:
        caps = caption_dict.get(img_id, [])
        eng_caps = eng_caption_dict.get(img_id, [])
        test_images.append(f"{image_path}{img_id}")
        if len(caps) < 5:
            print(img_id, len(caps))
        for c in caps:
            test_captions.append(c)
        for c in eng_caps:
            test_eng_captions.append(c)

    return test_images, test_captions,test_eng_captions

def get_test_images_for_xtd10(image_path,xtd10_data):
    test_images = []
    test_captions = []
    test_eng_captions = []
    if os.path.exists(image_path):
        for filename in os.listdir(image_path):
            # Create the full absolute path
            test_images.append(os.path.join(image_path, filename))
            row = xtd10_data.loc[xtd10_data['image_id'] == filename]
            test_captions.append(row['translated_caption_np'])
            test_eng_captions.append(row['caption'])

    test_images = []
    test_captions = []
    test_eng_captions = []

    for filename in os.listdir(image_path):
        # 1. Create and append the full image path
        full_path = os.path.join(image_path, filename)

        # 2. Find the corresponding row in the DataFrame
        # We use .iloc[0] to get the actual value from the first matching row found
        match = xtd10_data.loc[xtd10_data['image_id'] == filename, 'translated_caption_np']
        match_eng = xtd10_data.loc[xtd10_data['image_id'] == filename, 'caption']

        if not match.empty:
            test_images.append(full_path)
            test_captions.append(str(match.iloc[0]))
        else:
            print(f"Warning: No caption found for {filename}")

        if not match_eng.empty:
            test_eng_captions.append(str(match_eng.iloc[0]))
        else:
            print(f"Warning: No caption found for {filename}")

    return test_images, test_captions, test_eng_captions

def create_image_id_to_path_dict(image_path):
    """
    Creates a dictionary mapping base image IDs to their actual string file paths.
    e.g., {'ch_000023': 'dataset/xm3600/ch_000023.jpg'}
    """
    path_dict = {}

    if not os.path.exists(image_path):
        print(f"❌ Error: Directory '{image_path}' not found.")
        return path_dict

    # Filter for image files
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    files = [f for f in os.listdir(image_path) if f.lower().endswith(valid_extensions)]

    for filename in files:
        # Extract base ID (e.g., 'ch_000023.jpg' -> 'ch_000023')
        image_id = os.path.splitext(filename)[0]
        full_path = os.path.join(image_path, filename)

        path_dict[image_id] = full_path

    return path_dict

def get_test_images_for_xm3600(xm3600_data, image_path):
    """
    Aligns pre-loaded PIL images with their corresponding flattened target captions.

    Args:
        xm3600_data (pd.DataFrame): Dataframe containing 'image_id' and 'translated_caption_np'.
        image_id_with_image_dict (dict): Dictionary mapping 'image_id' to pre-loaded PIL Image objects.

    Returns:
        tuple: (list of PIL Images, list of flattened string captions)
    """
    # 1. Prepare and tokenize the captions split by '|'
    # Copying data to avoid SettingWithCopyWarning if xm3600_data is a slice
    image_id_with_image_dict = create_image_id_to_path_dict(image_path=image_path)
    caption_df = xm3600_data.copy()

    if isinstance(caption_df['translated_caption_np'].iloc[0], str):
        caption_df['translated_caption_np'] = caption_df['translated_caption_np'].str.split('|')

        def final_cleaner(text):
            # 1. Remove the outer brackets
            content = text.strip("[]")

            # 2. This Regex finds content between quotes that are followed by
            # either a newline, a space, or the end of the string.
            # It handles '...', "...", and ""...""
            pattern = r'["\'](.*?)(["\'])(?:\n|\s|$)'

            # Find all matches
            matches = re.findall(pattern, content)

            cleaned_list = []
            for text_content, closing_quote in matches:
                # Clean up the specific artifacts:
                # - Remove the escaped apostrophes (\')
                # - Fix the double-double quotes (""word"") to single ("word")
                c = text_content.replace("\\'", "'").replace('""', '"')
                # - Strip any remaining leading/trailing quotes
                c = c.strip("'\" ")
                if c:
                    cleaned_list.append(c)

            return cleaned_list

        caption_df['captions'] = caption_df['captions'].apply(final_cleaner)
    # 2. Map image_id to the list of captions
    invalid_captions_df = caption_df[caption_df['captions'].apply(lambda x: len(x) if isinstance(x, list) else 0) != 2]
    if len(invalid_captions_df) > 0:
        print("invalid_captions_df",invalid_captions_df)
    caption_dict_xm = {
        row['image_id']: {
            'nepali': row['translated_caption_np'],
            'english': row['captions']
        }
        for _, row in caption_df.iterrows()
    }

    test_images_xm = []
    test_captions_xm = []
    test_eng_captions_xm = []

    # 3. Align images and captions
    for img_id, caps in tqdm(caption_dict_xm.items(), desc="Aligning Images & Captions"):

        # Fetch the pre-processed PIL image safely
        pil_img = image_id_with_image_dict.get(img_id)

        if pil_img is None:
            # Fallback check: try removing file extension if key styles mismatch (.jpg vs no extension)
            img_id_clean = img_id.split('.')[0]
            pil_img = image_id_with_image_dict.get(img_id_clean)

        # 🚨 MISSING IMAGE CHECK: Skip entirely if the image is missing to prevent index shifting
        if pil_img is None:
            print(f"⚠️ Warning: Image {img_id} not found in image dictionary. Skipping.")
            continue

        nepali_list = caps['nepali']
        english_list = caps['english']
        # Validation check: Ensure both are lists and have exactly 2 elements
        if not isinstance(nepali_list, list) or len(nepali_list) != 2:
            print(f"⚠️ Warning: Expected 2 captions for {img_id}, got a mismatch.")
            continue

        # 1. Append the image EXACTLY ONCE
        test_images_xm.append(pil_img)

        # 2. Append the cleaned list of 2 Nepali captions as a single list entry
        test_captions_xm.extend([str(c).strip() for c in nepali_list])

        # 3. Append the cleaned list of 2 English captions as a single list entry
        test_eng_captions_xm.extend([str(c).strip() for c in english_list])

    return test_images_xm, test_captions_xm,test_eng_captions_xm

DATASET_LIST = ["FLICK8K","FLICK30K","XTD10","XM3600"]
# DATASET_LIST = ["FLICK8K"]
# DATASET_LIST = ["XM3600"]

def preprocess_all_datasets(clip_processor):
    processed_datasets = {}
    test_images = []
    test_captions = []
    test_eng_captions = []
    captions_per_image = 0
    for dataset_name in DATASET_LIST:
        if dataset_name == "FLICK8K":
            translated_dataset = pd.read_csv(FLICK_8K_CSV_FILE_NAME)
            test_img_ids = get_test_images("dataset/karpathy/dataset_flickr8k.json")
            image_path = "dataset/flickr8k/images/Flicker8k_Dataset/"
            test_images,test_captions,test_eng_captions = get_test_images_for_flickr(test_img_ids=test_img_ids,image_path=image_path,flickr_dataset=translated_dataset)
            captions_per_image = 5
        elif dataset_name == "FLICK30K":
            translated_dataset = pd.read_csv(FLICK_30K_CSV_FILE_NAME)
            test_img_ids = get_test_images("dataset/karpathy/dataset_flickr30k.json")
            image_path = "dataset/flickr30k/images/"
            test_images,test_captions,test_eng_captions = get_test_images_for_flickr(test_img_ids=test_img_ids,image_path=image_path,flickr_dataset=translated_dataset)
            captions_per_image = 5
        elif dataset_name == "XTD10":
            translated_dataset = pd.read_csv(XTD10_CSV_FILE_NAME)
            image_path = "dataset/xtd10/"
            test_images,test_captions,test_eng_captions = get_test_images_for_xtd10(image_path=image_path,xtd10_data=translated_dataset)
            captions_per_image = 1
        elif dataset_name == "XM3600":
            translated_dataset = pd.read_csv(XM3600_CSV_FILE_NAME)
            image_path = "dataset/xm3600/images/"
            test_images,test_captions,test_eng_captions = get_test_images_for_xm3600(xm3600_data=translated_dataset,image_path=image_path)
            captions_per_image = 2
        else:
            print(f"Dataset {dataset_name} not found.")
        if len(test_images)>0 and len(test_captions)>0:
            print(f"Preprocessing dataset {dataset_name}")
            print(f"test_images {len(test_images)}")
            print(f"test_captions {len(test_captions)}")
            dataset = EvaluationDataset(image_paths=test_images, captions=test_captions,eng_captions=test_eng_captions, image_processor=clip_processor)
            image_values, text_inputs,eng_text_inputs = dataset.get_all_data()
            processed_datasets[dataset_name] = {
                "image_values": image_values,
                "text_inputs": text_inputs,
                "eng_text_inputs": eng_text_inputs,
                "captions_per_image":captions_per_image
            }
        else:
            print(f"Skipping dataset {dataset_name} as number of images is {len(test_images)} and captions is {len(test_captions)}")
    return processed_datasets