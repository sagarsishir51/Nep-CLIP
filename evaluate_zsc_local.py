import os
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import CLIPProcessor
from dotenv import load_dotenv

from load_model import get_all_models_with_loaded_instance, TEACHER_ID
from dataset_utils import get_device
from EvaluationDataset import EvaluationDataset
from zsc_dataset_utils import local_dataset_mapper

load_dotenv()
BATCH_SIZE = 256
TEMPERATURE = 0.05
IMG_FOLDER = "dataset/nepali_dataset_zsc"
MAPPING_FILE = os.path.join(IMG_FOLDER, "_image_label_mapping.csv")
OUTPUT_FILE = "local_zsc_predictions.csv"
TOP_K_LIST = [1, 5, 10]


def predict(model, name, image_values, text_inputs, eng_text_inputs, gt_labels, device, batch_size,
            top_k=max(TOP_K_LIST)):
    img_emb = model.get_image_features(image_values=image_values, batch_size=batch_size, device=device)
    texts = eng_text_inputs if name == "CLIP-English" else text_inputs
    txt_emb = model.get_text_features(texts=texts, batch_size=batch_size, device=device)

    # Same scoring as compute_zsc_metrics
    img_norm = F.normalize(img_emb.to(device), p=2, dim=-1)
    txt_norm = F.normalize(txt_emb.to(device), p=2, dim=-1)
    probs = F.softmax((img_norm @ txt_norm.T) / TEMPERATURE, dim=1)
    top_probs, top_indices = torch.topk(probs, k=min(top_k, txt_norm.size(0)), dim=1)

    # 1-based rank of the actual class among all classes, as used for MRR in compute_zsc_metrics
    gt_tensor = torch.as_tensor(gt_labels).to(device)
    ranks = (probs.argsort(dim=1, descending=True) == gt_tensor.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
    return top_probs.cpu(), top_indices.cpu(), ranks.cpu()


def build_rows(name, mapped_data, top_probs, top_indices, ranks):
    labels_ne = mapped_data["text_inputs"]
    labels_en = mapped_data["eng_text_inputs"]
    types = mapped_data["types"]
    rows = []
    for i, (path, actual) in enumerate(zip(mapped_data["image_paths"], mapped_data["gt_labels"])):
        predicted = top_indices[i, 0].item()
        ranked = top_indices[i].tolist()
        row = {
            "model": name,
            "filename": os.path.basename(path),
            "actual_class": actual,
            "actual_label_en": labels_en[actual],
            "actual_label_ne": labels_ne[actual],
            "actual_type": types[actual],
            "predicted_class": predicted,
            "predicted_label_en": labels_en[predicted],
            "predicted_label_ne": labels_ne[predicted],
            "predicted_type": types[predicted],
            "confidence": round(top_probs[i, 0].item(), 4),
            "correct": predicted == actual,
            "actual_rank": ranks[i].item(),
            "reciprocal_rank": round(1.0 / ranks[i].item(), 4),
        }
        for k in TOP_K_LIST:
            row[f"in_top_{k}"] = actual in ranked[:k]
        # Ranked predictions up to the largest k; the first 5 entries are the top-5
        k = max(TOP_K_LIST)
        row[f"top_{k}_classes"] = " | ".join(str(j) for j in ranked)
        row[f"top_{k}_labels_en"] = " | ".join(labels_en[j] for j in ranked)
        row[f"top_{k}_labels_ne"] = " | ".join(labels_ne[j] for j in ranked)
        row[f"top_{k}_types"] = " | ".join(types[j] for j in ranked)
        row[f"top_{k}_confidences"] = " | ".join(f"{p:.4f}" for p in top_probs[i].tolist())
        rows.append(row)
    return rows


def compute_metrics(df):
    """Top-k accuracy, MRR and macro-F1 (all x100) for a group of prediction rows, as in compute_zsc_metrics.

    Scaled x100 to match how log_zsc_report prints them in evaluate_zsc.py.
    """
    metrics = {"images": len(df)}
    for k in TOP_K_LIST:
        metrics[f"top-{k} (%)"] = round(df[f"in_top_{k}"].mean() * 100, 2)
    metrics["MRR"] = round((1.0 / df["actual_rank"]).mean() * 100, 2)
    metrics["macro-F1"] = round(f1_score(df["actual_class"], df["predicted_class"], average="macro",
                                         zero_division=0) * 100, 2)
    return metrics


def print_metrics(title, rows):
    metrics = compute_metrics(pd.DataFrame(rows))
    parts = [f"top-{k} {metrics[f'top-{k} (%)']:.2f}%" for k in TOP_K_LIST]
    parts += [f"MRR {metrics['MRR']:.2f}", f"macro-F1 {metrics['macro-F1']:.2f}"]
    print(f"{title} ({metrics['images']} images): " + "  ".join(parts))


def print_model_summary(predictions):
    """Print one table comparing top-k accuracy (%), MRR and macro-F1 of every model, overall and per type."""
    # "All" row first, then one row per type, kept together for each model
    records = []
    for model_name, model_rows in predictions.groupby("model", sort=False):
        records.append({"model": model_name, "type": "All", **compute_metrics(model_rows)})
        for type_name, type_rows in model_rows.groupby("actual_type"):
            records.append({"model": model_name, "type": type_name, **compute_metrics(type_rows)})
    summary = pd.DataFrame(records).set_index(["model", "type"])
    print("\n=================================================================== Summary ===================================================================")
    print(summary.to_string())


def main():
    parser = argparse.ArgumentParser(description="Log per-image zero-shot predictions on the local Nepali dataset")
    parser.add_argument("--mapping", default=MAPPING_FILE)
    parser.add_argument("--images", default=IMG_FOLDER)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    device = get_device()
    clip_processor = CLIPProcessor.from_pretrained(TEACHER_ID)
    model_configs = get_all_models_with_loaded_instance(device=device)

    mapped_data = local_dataset_mapper(mapping_file=args.mapping, img_folder=args.images)
    dataset = EvaluationDataset(image_paths=mapped_data["image_paths"], test_texts=mapped_data["text_inputs"],
                                test_eng_texts=mapped_data["eng_text_inputs"], image_processor=clip_processor,
                                use_templates=True)
    image_values, text_inputs, eng_text_inputs = dataset.get_all_data()

    all_rows = []
    for name, model in model_configs.items():
        print(f"=================================================================== {name} ===================================================================")
        top_probs, top_indices, ranks = predict(model, name, image_values, text_inputs, eng_text_inputs,
                                                mapped_data["gt_labels"], device, BATCH_SIZE)
        rows = build_rows(name, mapped_data, top_probs, top_indices, ranks)
        print_metrics(name, rows)
        for type_name in sorted({r["actual_type"] for r in rows}):
            print_metrics(f"  {type_name}", [r for r in rows if r["actual_type"] == type_name])
        all_rows.extend(rows)
        torch.cuda.empty_cache()

    predictions = pd.DataFrame(all_rows)
    predictions.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Saved {len(all_rows)} predictions to {args.output}")
    print_model_summary(predictions)


if __name__ == "__main__":
    main()
