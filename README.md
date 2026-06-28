# Nep-CLIP 🇳🇵

Nep-CLIP is a Nepali vision-language model project inspired by CLIP, designed to bridge images and Nepali text through multimodal learning. This repository contains the complete codebase, translated datasets, pretrained models, and training/evaluation pipelines used for Nepali image-text understanding tasks.

---

## 📊 Datasets

The processed Nepali image-text pairs are available for public use.

> 🔗 <a href="https://drive.google.com/drive/folders/1kovgPWgzIfNO_Cv64rGTgCTBQDelE8k2?usp=sharing" target="_blank" rel="noopener noreferrer"><b>Download Nep-CLIP Datasets (Google Drive)</b></a>

---

## 📂 Dataset File Descriptions

The dataset is divided into training/validation splits and benchmark evaluation datasets.

### 📁 Training Data

Contains high-volume image-text pairs used for Knowledge Distillation (KD).

* **CC3M Parquet Files (568 files)**
* **MSCOCO Parquet File (1 file)**

**Format:** Apache Parquet (`.parquet`)

---

### 📁 Validation Data

Contains validation splits for monitoring model performance during training.

* **CC3M Parquet Files (3 files)**
* **MSCOCO Parquet File (1 file)**

**Format:** Apache Parquet (`.parquet`)

---

### 📁 Translated Benchmark Datasets

Evaluation datasets translated into Nepali using Google Translate and IndicTrans2.

| Dataset   | File                                           |
| --------- | ---------------------------------------------- |
| Flickr8k  | `flickr8k_nepali_output_google_translate.csv`  |
| Flickr30k | `flickr30k_nepali_output_google_translate.csv` |
| XTD10     | `xtd10_google_translate.csv`                   |
| XM3600    | `xm3600_google_translate.csv`                  |

---

## 🧠 Pretrained Models

The distilled Nep-CLIP checkpoints are publicly available.

> 🔗 <a href="https://drive.google.com/drive/u/1/folders/1ZY_gS27NwRj5RoZFb-yCj9kh-i09WjCh?usp=sharing" target="_blank" rel="noopener noreferrer"><b>Download Nep-CLIP Models (Google Drive)</b></a>

---

### 1. Nep-CLIP (RoBERTa)

**Checkpoint**

```
RoBERTa_Nepali_110M_distillation_epoch_4_step_150000.pt
```

**Description**

A distilled Nep-CLIP model using a 110M parameter Nepali RoBERTa encoder trained through knowledge distillation.

---

### 2. Nep-CLIP (BERT)

**Checkpoint**

```
Shushant-nepaliBERT_distillation_epoch_4_step_150000.pt
```

**Description**

A distilled Nep-CLIP model using the Shushant NepaliBERT encoder trained through knowledge distillation.

---

## 📋 Evaluated Models

The following models were evaluated for image-text retrieval:

* Nep-CLIP (RoBERTa)
* Nep-CLIP (BERT)
* NLLB-CLIP
* M-CLIP
* CLIP-English
* CLIP-English (Nepali Input)

---
# 📈 Benchmark Results

This section summarizes the performance of **Nep-CLIP** across multiple downstream evaluation tasks.

> All benchmark evaluations were conducted on **Google Colab (NVIDIA L4 GPU)**.

---

## Image-Text Retrieval
The following figure summarizes the image-text retrieval performance of Nep-CLIP across all evaluation benchmarks (Flickr8k, Flickr30k, XTD10, and XM3600).

<p>
  <img src="assets/nepclip_image_retrieval_benchmark_results.png"
       alt="Nep-CLIP IR Benchmark Results"
       width="1156">
</p>

**Figure:** Cross-modal retrieval performance on the Nepali datasets. **w/ PP** denotes models trained with preprocessing. **Bold** values indicate the best performance among the distilled and multilingual models, excluding the English upper-bound (CLIP-English*).

## Zero-Shot Image Classification
The following figure summarizes the zero shot classification performance of Nep-CLIP across all evaluation benchmarks (Babel-ImageNet Nepali, CIFAR10, CIFAR100).

<p>
  <img src="assets/nepclip_zsc_benchmark_results.png"
       alt="Nep-CLIP ZSC Benchmark Results"
       width="1118">
</p>

**Figure:** Zero-shot classification performance on the Nepali image datasets across different benchmarks. Top-1 accuracy, Top-5 accuracy, MRR, and MAF1 are reported. **Bold** values indicate the best performance among the distilled and multilingual models, excluding the English upper-bound (CLIP-English*).

## 📌 Key Findings

* 🥇 **Nep-CLIP (RoBERTa)** consistently achieves the best performance among Nepali models across all benchmarks.
* 🥈 **Nep-CLIP (BERT)** provides competitive performance and closely follows the RoBERTa model.
* 🌍 Both Nep-CLIP models significantly outperform multilingual baselines (**NLLB-CLIP** and **M-CLIP**) on Nepali image-text retrieval.
* 📈 **CLIP-English*** serves as the English upper-bound, while directly applying English CLIP to Nepali text without adaptation leads to near-zero retrieval performance.

---

# 🛠️ Setup and Installation

## Prerequisites

* Python 3.11
* Git
* pip

---

## 1. Clone the Repository

```bash
git clone https://github.com/<username>/Nep-CLIP.git
cd Nep-CLIP
```

---

## 2. Create a Virtual Environment

### Linux / macOS

```bash
python3.11 -m venv venv_nep_clip
source venv_nep_clip/bin/activate
```

### Windows

```bash
python3.11 -m venv venv_nep_clip
venv_nep_clip\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download Evaluation Datasets

```bash
python3 download_dataset.py [-d {flickr8k,flickr30k,xtd10,xm3600,cifar10,cifar100,all}]
```

Examples:

```bash
python3 download_dataset.py
python3 download_dataset.py -d flickr8k
python3 download_dataset.py -d xtd10
```
> **Note**
> - The script automatically downloads and prepares the supported evaluation datasets listed above.
> - **ImageNet** is **not** downloaded automatically due to its licensing and access restrictions. To evaluate on ImageNet, download the following files from **ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)**:
>   - **Development kit (Task 1 & 2)**
>   - **Validation images (all tasks)**
> - Download them from the official ImageNet website: https://www.image-net.org/download.php
> - After downloading, extract and place them in the following directory structure:
>
> ```text
> dataset/
> └── imagenet/
>     ├── ILSVRC2012_devkit_t12/
>     └── ILSVRC2012_img_val/
> ```
>
> Ensure the folder names match exactly as shown above.
## 5. Run Image Retrieval Evaluation

```bash
python3 evaluate_image_retrieval.py
```
## 5. Run Zero-Shot Image Classification

```bash
python3 evaluate_zsc.py
```
---

## ⚠️ NLLB & M-CLIP Compatibility

The repository installs:

```text
transformers==5.12.1
```

To evaluate **NLLB-CLIP** or **M-CLIP**, install:

```bash
pip uninstall transformers -y
pip install transformers==4.38.2
```

---

## 📝 Notes

* Activate the virtual environment before running scripts.
* Reinstall dependencies after updating `requirements.txt`.
* Exit the environment using:

```bash
deactivate
```

---