"""
Sample 80 pairs per dataset for the human translation-validation study.
Keeps trace-back IDs (row `id` where present + `image_id`) + English + Nepali.
Output columns: pair_id, dataset, id, image_id, english, nepali
"""
import ast
import pandas as pd
import numpy as np

from dataset_utils import (
    FLICK_8K_CSV_FILE_NAME,
    FLICK_30K_CSV_FILE_NAME,
    XTD10_CSV_FILE_NAME,
    XM3600_CSV_FILE_NAME,
    TRAINING_DATA_FILE_NAME,
)

SEED = 42
rng = np.random.default_rng(SEED)
PER_DATASET = 60

FILES = {
    "training":  {"path": TRAINING_DATA_FILE_NAME, "en": "text",     "np": "translated_text"},
    "flickr8k":  {"path": FLICK_8K_CSV_FILE_NAME,  "en": "caption",  "np": "translated_caption_np"},
    "flickr30k": {"path": FLICK_30K_CSV_FILE_NAME, "en": "caption",  "np": "translated_caption_np"},
    "xtd10":     {"path": XTD10_CSV_FILE_NAME,      "en": "caption",  "np": "translated_caption_np"},
    "xm3600":    {"path": XM3600_CSV_FILE_NAME,     "en": "captions", "np": "translated_caption_np"},
}


def load(path):
    path = str(path)
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def first_caption(x):
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) else ""
    s = str(x)
    if s.startswith("[") and ("'" in s or '"' in s):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)) and v:
                return v[0]
        except Exception:
            pass
    return s


parts = []
for dset, cfg in FILES.items():
    df = load(cfg["path"])
    for role, col in (("en", cfg["en"]), ("np", cfg["np"])):
        if col not in df.columns:
            raise SystemExit(f"{dset}: column '{col}' ({role}) not found. Present: {df.columns.tolist()}")

    english = df[cfg["en"]].map(first_caption) if dset == "xm3600" else df[cfg["en"]]
    out = pd.DataFrame({
        "id":       df["id"].values if "id" in df.columns else "",
        "image_id": df["image_id"].values if "image_id" in df.columns else df.index,
        "english":  english.values,
        "nepali":   df[cfg["np"]].values,
    })
    out["dataset"] = dset

    n = min(PER_DATASET, len(out))
    if len(out) < PER_DATASET:
        print(f"  WARNING: {dset} has only {len(out)} rows; taking all {n}.")
    idx = rng.choice(out.index.values, size=n, replace=False)
    parts.append(out.loc[idx])
    print(f"{dset}: {len(df)} rows -> sampled {n}")

sample = pd.concat(parts, ignore_index=True)
sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
sample.insert(0, "pair_id", [f"E{i:04d}" for i in range(1, len(sample) + 1)])
sample = sample[["pair_id", "dataset", "id", "image_id", "english", "nepali"]]

sample.to_csv("sample_pairs.csv", index=False)
print(f"\nTotal sampled: {len(sample)}")
print(sample["dataset"].value_counts().to_string())
print("Raters see: english, nepali.  dataset + id + image_id are hidden trace-back keys.")
print("Wrote sample_pairs.csv")