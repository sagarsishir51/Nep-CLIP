# Nep-CLIP 🇳🇵

Nep-CLIP is a Nepali vision-language model project inspired by CLIP, designed to bridge images and Nepali text through multimodal learning. This repository contains the complete codebase, translated datasets, pretrained models, and training/evaluation pipelines used for Nepali image-text understanding tasks.

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
* **Flickr8k**
* **Flickr30k**
* **XTD10**
* **XM3600**

---

## 🧠 Pretrained Models

The distilled Nep-CLIP models are available for public use. 
> 🔗 <a href="https://drive.google.com/drive/u/1/folders/1ZY_gS27NwRj5RoZFb-yCj9kh-i09WjCh?usp=sharing" target="_blank" rel="noopener noreferrer">**Download Nep-CLIP Models (Google Drive)**</a>

---
The following distilled checkpoint files are hosted in the Google Drive directory under the model artifacts section:

### 1. `RoBERTa_Nepali_110M distillation_epoch_4_step_150000.pt`
* **Description:** Pretrained PyTorch model checkpoint utilizing a 110M parameter Nepali RoBERTa architecture as the student text encoder. Aligned via knowledge distillation at Epoch 4, Step 150,000.

### 2. `Shushant-nepaliBERT distillation_epoch_4_step_150000.pt`
* **Description:** Pretrained PyTorch model checkpoint utilizing the Shushant-nepaliBERT architecture as the student text encoder. Aligned via knowledge distillation at Epoch 4, Step 150,000.

---

## 📓 Notebooks & Execution Order

Below is the structured list of Jupyter notebooks included in this repository to replicate the translation, training, and evaluation pipelines:

### 1. `Translate_with_IndicTrans2.ipynb`
* **Description:** The text translation pipeline that uses the `IndicTrans2` model to translate English image captions into high-quality Nepali text datasets.

### 2. `Preprocessed_CC3M_and_MSCOCO_with_Val_loss_Feature_Distillation.ipynb`
* **Description:** Handles data preprocessing (PP) and runs the main Knowledge Distillation (KD) training loop. It transfers vision-language features from the teacher model to the Nepali student model while monitoring validation loss.

### 3. `Evaluation_of_Image_Retrieval_Task.ipynb`
* **Description:** Benchmarking pipeline for Cross-Modal Retrieval. It evaluates the trained model on Image-to-Text and Text-to-Image retrieval tasks using metrics like Recall@K and MRR.

### 4. `Evaluation_of_ZSC_Task.ipynb`
* **Description:** Evaluation pipeline for Zero-Shot Classification (ZSC). Tests the model's out-of-domain classification capability across different Nepali text prompts and datasets.