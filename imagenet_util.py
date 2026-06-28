import scipy.io
import requests
import os
import json


def imagenet_to_babel_mapper(gt_file,mat_file,img_folder):
    # 1. Load the metadata
    meta = scipy.io.loadmat(mat_file, squeeze_me=True)
    synsets = meta['synsets']

    # 2. Extract the 1,000 target classes (IDs 1 to 1000)
    low_level = []
    for s in synsets:
        if s[0] <= 1000:
            low_level.append({'ID': int(s[0]), 'WNID': s[1], 'Label': s[2]})

    # 3. Sort them alphabetically by WNID to get the 0-999 order
    sorted_by_wnid = sorted(low_level, key=lambda x: x['WNID'])

    # 4. Create Conversion Dictionaries
    # From Ground Truth ID (1-1000) -> Model Index (0-999)
    id_to_model_idx = {item['ID']: i for i, item in enumerate(sorted_by_wnid)}

    # From Model Index (0-999) -> Ground Truth ID (1-1000)
    model_idx_to_id = {i: item['ID'] for i, item in enumerate(sorted_by_wnid)}

    # Local path to the JSON file
    json_path = "assets/babel_imagenet.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nep_indices, nep_labels = data["NE"]

    imagenet_idx_to_nepali = {model_idx_to_id[idx]: label for idx, label in zip(nep_indices, nep_labels)}

    print(f"Loaded {len(nep_labels)} Nepali labels.")

    images = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.jpeg', '.jpg', '.JPEG'))])

    with open(gt_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]

    all_matches = {
        img: {
            "idx": lbl,
            "label_en": sorted_by_wnid[id_to_model_idx[lbl]]["Label"]
        }
        for img, lbl in zip(images, labels)
    }
    supported_indices = imagenet_idx_to_nepali.keys()
    filtered_matches = {
        img: {
            "idx": info["idx"],
            "label_ne": imagenet_idx_to_nepali[info["idx"]],
            "label_en": info["label_en"],
            "path": os.path.abspath(os.path.join(img_folder, img))
        }
        for img, info in all_matches.items()
        if info["idx"] in supported_indices
    }

    image_paths = [info["path"] for info in filtered_matches.values()]
    total_labels_ne = [info["label_ne"] for info in filtered_matches.values()]
    total_labels_en = [info["label_en"] for info in filtered_matches.values()]

    text_prompts = sorted(list(set(total_labels_ne)))
    label_to_idx = {label: i for i, label in enumerate(text_prompts)}
    gt_labels = [label_to_idx[label] for label in total_labels_ne]
    text_prompts_eng = sorted(list(set(total_labels_en)))
    label_to_idx_en = {label: i for i, label in enumerate(text_prompts_eng)}
    gt_labels_en = [label_to_idx_en[label] for label in total_labels_en]

    return {
        "image_paths":image_paths,
        "gt_labels":gt_labels,
        "gt_labels_en":gt_labels_en,
        "text_inputs":text_prompts,
        "eng_text_inputs":text_prompts_eng
    }
