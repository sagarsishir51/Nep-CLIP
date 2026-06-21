# Nep-CLIP 🇳🇵

Nep-CLIP is a Nepali vision-language model project inspired by CLIP, designed to bridge images and Nepali text through multimodal learning. This repository contains the complete codebase, translated datasets, pretrained models, and training/evaluation pipelines used for Nepali image-text understanding tasks.

---

## 📊 Datasets

The processed Nepali image-text pairs are available for public use. 
> 🔗 <a href="https://drive.google.com/drive/folders/1kovgPWgzIfNO_Cv64rGTgCTBQDelE8k2?usp=sharing" target="_blank" rel="noopener noreferrer">**Download Nep-CLIP Datasets (Google Drive)**</a>

---

## 📂 Dataset File Descriptions

The dataset is divided into training/validation splits and specific benchmark files.

### 📁 Training Data
*Contains high-volume data for the Knowledge Distillation (KD) process.*

#### `CC3M Parquet Files (568 files) and MSCOCO Parquet File (1 file)`
* **Format:** Apache Parquet (`.parquet`)

---

### 📁 Validation Data
*Contains standard splits for monitoring model performance during training.*

#### `CC3M Parquet Files (3 files) and MSCOCO Parquet File (1 file)`
* **Format:** Apache Parquet (`.parquet`)

---

### 📁 Translated Benchmark Datasets
*Diverse evaluation datasets translated using Google Translate and IndicTrans2.*
* **Flickr8k**: flickr8k_nepali_output_google_translate.csv
* **Flickr30k**: flickr30k_nepali_output_google_translate.csv
* **XTD10**: xm3600_google_translate.csv
* **XM3600**: xtd10_google_translate.csv

---

## 🧠 Pretrained Models

The distilled Nep-CLIP models are available for public use. 
> 🔗 <a href="https://drive.google.com/drive/u/1/folders/1ZY_gS27NwRj5RoZFb-yCj9kh-i09WjCh?usp=sharing" target="_blank" rel="noopener noreferrer">**Download Nep-CLIP Models (Google Drive)**</a>

---
The following distilled checkpoint files are hosted in the Google Drive directory under the model artifacts section:

### 1. `Nep-CLIP Model (RoBERTa based)`
* **File Name:** RoBERTa_Nepali_110M distillation_epoch_4_step_150000.pt
* **Description:** Pretrained PyTorch model checkpoint utilizing a 110M parameter Nepali RoBERTa architecture as the student text encoder. Aligned via knowledge distillation at Epoch 4, Step 150,000.

### 2. `Nep-CLIP Model (BERT based)`
* **File Name:** Shushant-nepaliBERT distillation_epoch_4_step_150000.pt
* **Description:** Pretrained PyTorch model checkpoint utilizing the Shushant-nepaliBERT architecture as the student text encoder. Aligned via knowledge distillation at Epoch 4, Step 150,000.

---

## 📋 Evaluated Models

The following models were evaluated for image-text retrieval and zero-shot classification:

- **BERT** (Shushant-nepaliBERT student encoder)
- **RoBERTa** (Nepali RoBERTa student encoder)
- **NLLB**
- **MCLIP**
- **CLIP (English)**
- **CLIP (Nepali input)**

---
# 🛠️ Setup and Installation Guide

Follow these steps to set up the development environment and reproduce the evaluation results.

---

## Prerequisites

- **Python 3.11** — verify with `python --version`
- **Git** — to clone the repository
- **pip** — comes bundled with Python 3.9+

---

## Step 2: Create and Activate a Virtual Environment

Create the environment using **Python 3.11** specifically.

**Check if Python 3.11 is installed:**
```bash
python3 --version
```

If this shows a version other than 3.11, follow the steps below to install Python 3.11 specifically.

**If not installed:**

**Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

**macOS (using Homebrew)**
```bash
brew install python@3.11
```

**Windows**
Download the installer from the [official Python website](https://www.python.org/downloads/release/python-3110/) and run it. Make sure to check **"Add Python to PATH"** during installation.

**Create and activate the virtual environment:**

**Linux / macOS**
```bash
python3.11 -m venv venv_nep_clip
source venv_nep_clip/bin/activate
```

**Windows**
```bash
python3.11 -m venv venv_nep_clip
venv_nep_clip\Scripts\activate
```

> You should see `(venv_nep_clip)` in your terminal prompt once the environment is active.

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```
---
## Step 4: Download Required Datasets

```bash
python3 download_dataset.py [-d {flickr8k,flickr30k,xtd10,xm3600,all}]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-d` | `all` | Dataset to download |

**Examples:**
```bash
python3 download_dataset.py            # download all datasets
python3 download_dataset.py -d flickr8k  # Flickr8k only
python3 download_dataset.py -d xtd10     # XTD10 only
```

Ensure you have a stable internet connection and sufficient disk space before running.

## Step 5: Run Image Retrieval Evaluation

```bash
python3 evaluate_image_retrieval.py
```

Results will be printed to the console, depending on the script configuration.

---
> ⚠️ **Disclaimer (NLLB & MCLIP):** By default, `transformers==5.12.1` is installed. To use the NLLB and MCLIP models, you need `transformers==4.38.2` instead:
> ```bash
> pip uninstall transformers -y
> pip install transformers==4.38.2
> ```
## Notes

- Always activate the virtual environment before running any scripts.
- If you update dependencies, re-run `pip install -r requirements.txt`.
- To deactivate the virtual environment when done: `deactivate`