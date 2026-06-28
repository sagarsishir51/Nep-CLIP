import json
import os

import torch
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torchvision

from EvaluationDataset import EvaluationDataset
from imagenet_util import imagenet_to_babel_mapper
from label_mapper import CIFAR10_LABELS_NE, CIFAR10_LABELS_EN, CIFAR100_LABELS_NE, \
    CIFAR100_LABELS_EN

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

DATASET_LIST = ["CIFAR10","CIFAR100","IMAGENET"]


def get_test_images_for_cifar100(image_path,xtd10_data):
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


def preprocess_zsc_datasets(clip_processor):
    processed_datasets = {}
    for dataset_name in DATASET_LIST:
        test_images = []
        test_labels = []
        gt_labels = []
        test_eng_labels = []
        gt_labels_en=[]
        if dataset_name == "CIFAR10":
            try:
                test_labels = CIFAR10_LABELS_NE
                test_eng_labels = CIFAR10_LABELS_EN
                cifar10_dataset = torchvision.datasets.CIFAR10(root='dataset/cifar10', train=False, download=False)
                gt_labels = cifar10_dataset.targets
                for i in range(len(cifar10_dataset)):
                    img, _ = cifar10_dataset[i]
                    test_images.append(img)
            except Exception as e:
                print(e)
                print("CIFAR-10 dataset not found. First download dataset using script download_dataset.py")
                continue
        elif dataset_name == "CIFAR100":
            try:
                test_labels =CIFAR100_LABELS_NE
                test_eng_labels = CIFAR100_LABELS_EN
                cifar100_dataset = torchvision.datasets.CIFAR100(root='dataset/cifar100', train=False, download=False)
                gt_labels = cifar100_dataset.targets
                for i in range(len(cifar100_dataset)):
                    img, _ = cifar100_dataset[i]
                    test_images.append(img)
            except Exception as e:
                print(e)
                print("CIFAR-100 dataset not found. First download dataset using script download_dataset.py")
                continue
        elif dataset_name == "IMAGENET":
            try:
                gt_file = "dataset/imagenet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
                mat_file = "dataset/imagenet/ILSVRC2012_devkit_t12/data/meta.mat"
                img_folder = "dataset/imagenet/ILSVRC2012_img_val"
                mapped_data = imagenet_to_babel_mapper(gt_file=gt_file, mat_file=mat_file, img_folder=img_folder)
                gt_labels_en = mapped_data["gt_labels_en"]
                gt_labels = mapped_data["gt_labels"]
                test_labels = mapped_data["text_inputs"]
                test_eng_labels = mapped_data["eng_text_inputs"]
                test_images = mapped_data["image_paths"]
            except Exception as e:
                print(e)
                print("Imagenet dataset not found. First download dataset and configure it with the help of README file")
                continue
        else:
            print(f"Dataset {dataset_name} not found.")
        if len(test_images)>0 and len(test_labels)>0 and len(gt_labels)>0:
            print(f"Preprocessing dataset {dataset_name}")
            dataset = EvaluationDataset(image_paths=test_images, test_texts=test_labels,test_eng_texts=test_eng_labels, image_processor=clip_processor,use_templates=True)
            image_values, text_inputs,eng_text_inputs = dataset.get_all_data()
            processed_datasets[dataset_name] = {
                "image_values": image_values,
                "text_inputs": text_inputs,
                "eng_text_inputs": eng_text_inputs,
                "gt_labels":gt_labels,
                "gt_labels_en": gt_labels_en

            }
        else:
            print(f"Skipping dataset {dataset_name} as number of images is {len(test_images)}, labels is {len(test_labels)} and ground truth is {len(gt_labels)}")
    return processed_datasets