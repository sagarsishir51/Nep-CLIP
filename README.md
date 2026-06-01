# Nep-CLIP 🇳🇵

> 🚧 **Coming Soon**  
> Full codebase, pretrained models, training pipelines, and documentation will be released soon.

Nep-CLIP is a Nepali vision-language model project inspired by CLIP, designed to bridge images and Nepali text through multimodal learning. This repository contains the complete codebase, translated datasets, pretrained models, and training/evaluation pipelines used for Nepali image-text understanding tasks.

## 📊 Datasets

The processed Nepali image-text pairs are available for public use. 
> 🔗 <a href="https://drive.google.com/drive/folders/1kovgPWgzIfNO_Cv64rGTgCTBQDelE8k2?usp=sharing" target="_blank" rel="noopener noreferrer">**Download Nep-CLIP Datasets (Google Drive)**</a>

## 📂 Dataset File Descriptions

The dataset is divided into training/validation splits and specific benchmark files.

---

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

###  Translated Data of Different datasets using Google Translate and indictrans2 
* **Flickr8k**
* **Flickr30k**
* **XTD10**
* **XM3600**

