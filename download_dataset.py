import argparse
import math
import shutil
import threading
import urllib.parse
import urllib.request
import zipfile
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import torchvision
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from datasets import load_dataset

load_dotenv()
def download_chunk(url, start, end, chunk_id, tmp_dir, pbar, lock):
    """Download a specific byte range of a file and update progress smoothly."""
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={start}-{end}")

    chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_id:04d}")

    with urllib.request.urlopen(req) as response:
        with open(chunk_path, "wb") as f:
            # Read in small 64KB increments rather than loading the whole chunk at once
            buffer_size = 64 * 1024
            while True:
                buffer = response.read(buffer_size)
                if not buffer:
                    break
                f.write(buffer)

                # Update the progress bar smoothly as data streams in
                with lock:
                    pbar.update(len(buffer))

    return chunk_id


def fast_download(url, dest_path, num_threads=8):
    """Download a file in parallel chunks, with a smooth progress bar."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        accept_ranges = response.headers.get("Accept-Ranges", "none")

    if accept_ranges == "none" or total_size == 0:
        print("  Server doesn't support parallel download, falling back to single-thread...")
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            def reporthook(count, block_size, total):
                pbar.update(block_size)

            urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        return

    chunk_size = math.ceil(total_size / num_threads)
    tmp_dir = dest_path + ".parts"
    os.makedirs(tmp_dir, exist_ok=True)

    print(
        f"  Downloading {os.path.basename(dest_path)} ({total_size / 1024 ** 2:.1f} MB) with {num_threads} threads...")

    # Define the lock and progress bar out here so threads can share it safely
    lock = threading.Lock()
    with tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(dest_path)) as pbar:

        chunks = [
            (i, i * chunk_size, min((i + 1) * chunk_size - 1, total_size - 1))
            for i in range(num_threads)
        ]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Pass pbar and lock directly into the streaming download function
            futures = [
                executor.submit(download_chunk, url, start, end, chunk_id, tmp_dir, pbar, lock)
                for chunk_id, start, end in chunks
            ]
            for f in as_completed(futures):
                f.result()

    # Merge chunks
    with open(dest_path, "wb") as out:
        for i in range(num_threads):
            chunk_path = os.path.join(tmp_dir, f"chunk_{i:04d}")
            with open(chunk_path, "rb") as cf:
                out.write(cf.read())
            os.remove(chunk_path)
    os.rmdir(tmp_dir)


def extract_with_progress(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        with tqdm(
            total=len(members),
            desc=f"Extracting {os.path.basename(zip_path)}",
            unit="file",
        ) as pbar:
            for member in members:
                zf.extract(member, extract_to)
                pbar.update(1)


def download_flickr8k_dataset(num_threads=8):
    base_dir = "dataset/flickr8k"
    os.makedirs(base_dir, exist_ok=True)
    base_url = (
        "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k"
    )

    downloads = [
        {
            "name": "Images",
            "url": f"{base_url}/Flickr8k_Dataset.zip",
            "zip": f"{base_dir}/Flickr8k_Dataset.zip",
            "check": f"{base_dir}/images",
            "extract_to": f"{base_dir}/images",
        },
        {
            "name": "Captions",
            "url": f"{base_url}/Flickr8k_text.zip",
            "zip": f"{base_dir}/Flickr8k_text.zip",
            "check": f"{base_dir}/Flickr8k.token.txt",
            "extract_to": base_dir,
        },
    ]
    for item in downloads:
        print(f"\n[{item['name']}]")
        if os.path.exists(item["check"]):
            print("  Already exists, skipping.")
            continue

        if not os.path.exists(item["zip"]):
            fast_download(item["url"], item["zip"], num_threads=num_threads)
        else:
            print("  Zip already downloaded, skipping fetch.")

        os.makedirs(item["extract_to"], exist_ok=True)
        extract_with_progress(item["zip"], item["extract_to"])
        os.remove(item["zip"])
        print("  Done.")

    print("\nFlickr8k dataset ready.")


def download_flickr30k_dataset():
    base_dir = "dataset/flickr30k"
    file_name = "flickr_annotations_30k.csv"
    images_extract_dir = f"{base_dir}/images"
    token_file = f"{base_dir}/{file_name}"

    os.makedirs(base_dir, exist_ok=True)
    print("\n[Images]")
    if not os.path.exists(images_extract_dir):
        print("  Downloading Flickr30k images from HuggingFace...")
        zip_path = hf_hub_download(
            repo_id="nlphuji/flickr30k",
            filename="flickr30k-images.zip",
            repo_type="dataset",
        )
        print("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            with tqdm(
                total=len(members), desc="Extracting images", unit="file"
            ) as pbar:
                for member in members:
                    member.filename = os.path.basename(member.filename)
                    if member.filename:
                        zf.extract(member, images_extract_dir)
                    pbar.update(1)
        print(f"  Flickr30k images ready at '{images_extract_dir}'.")
    else:
        print(f"  Images already exist at '{images_extract_dir}', skipping.")

    print("\n[Captions]")
    if not os.path.exists(token_file):
        print("  Downloading Flickr30k captions from HuggingFace...")
        caption_path = hf_hub_download(
            repo_id="nlphuji/flickr30k",
            filename=file_name,
            repo_type="dataset",
        )
        os.makedirs(os.path.dirname(token_file), exist_ok=True)
        shutil.copy(caption_path, token_file)
        print(f"  Flickr30k captions ready at '{token_file}'.")
    else:
        print(f"  Captions already exist at '{token_file}', skipping.")
    print("\nFlickr30k dataset ready.")


def download_karpathy_splits():
    """
    Downloads and extracts all Karpathy JSON splits directly into base_dir
    using a custom User-Agent header to prevent server download blocks.
    """
    base_dir = "dataset/karpathy"
    os.makedirs(base_dir, exist_ok=True)
    temp_zip = "caption_datasets.zip"
    url = "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"

    print("\n[Karpathy Splits]")

    # Only download if the target directory is currently empty
    if not os.listdir(base_dir):
        try:
            print("  Downloading Karpathy split archive...")

            # Mimic a real browser header so Stanford's server accepts the download request
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
            urllib.request.install_opener(opener)

            # Download the file
            urllib.request.urlretrieve(url, temp_zip)

            # Double-check that the file actually exists on your machine before unzipping
            if not os.path.exists(temp_zip):
                raise FileNotFoundError(f"Failed to download zip archive from server.")

            print(f"  Extracting files directly to '{base_dir}'...")
            with zipfile.ZipFile(temp_zip, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue

                    # Strip out the zip file's internal 'dataset/' directory wrapping
                    filename = os.path.basename(member.filename)
                    if not filename:
                        continue

                    target_path = os.path.join(base_dir, filename)

                    # Extract the raw data out to our flat base folder
                    with zf.open(member) as source, open(target_path, "wb") as target:
                        target.write(source.read())

            print(f"  Extraction complete. Files successfully located in '{base_dir}'.")

        except Exception as e:
            print(f"  An error occurred during Karpathy split setup: {e}")
        finally:
            # Clean up the zip file safely if it exists
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
    else:
        print(f"  Directory '{base_dir}' already has files. Skipping download.")

    print("  Current files in directory:", os.listdir(base_dir))


def download_xtd10_dataset():
    """
    Downloads the XTD10 image archive, creates the target folder structure
    if it doesn't exist, and extracts files directly into it.
    """
    url = "https://huggingface.co/datasets/Haon-Chen/XTD-10/resolve/main/XTD10_dataset.tar.gz"
    temp_tar = "xtd10.tar.gz"
    target_dir = "dataset/xtd10"
    print("\n[XTD10 Dataset Setup]")

    # 1. Ensure the custom destination directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Skip download if the folder already has the images
    if len(os.listdir(target_dir)) >= 1000:
        print(f"  Dataset already ready at '{target_dir}' ({len(os.listdir(target_dir))} images). Skipping.")
        return

    try:
        print("  Downloading pre-packaged XTD10 archive from HF Hub...")
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )

        with urllib.request.urlopen(req) as response, open(temp_tar, 'wb') as out_file:
            out_file.write(response.read())

        print(f"  Extracting archive contents directly to '{target_dir}'...")

        # 2. Extract files natively
        with tarfile.open(temp_tar, "r:gz") as tar:
            # If the tar file contains a root 'xtd10' directory, flatten it
            for member in tar.getmembers():
                if member.isfile():
                    # Clean the file path so it drops the top-level folder name inside the archive
                    filename = os.path.basename(member.name)
                    dest_path = os.path.join(target_dir, filename)

                    # Write the extracted file directly into your target directory
                    with tar.extractfile(member) as source, open(dest_path, "wb") as target:
                        shutil.copyfileobj(source, target)

        # 3. Validation Check
        image_count = len(os.listdir(target_dir))
        print(f"  Success! Total images collected in '{target_dir}': {image_count}")

    except Exception as e:
        print(f"  ❌ Error encountered during setup: {e}")

    finally:
        # 4. Clean up the temporary tar archive
        if os.path.exists(temp_tar):
            os.remove(temp_tar)


def download_xm3600_dataset():
    """
    Downloads the entire raw Crossmodal-3600 dataset repository directly
    to the target folder using the huggingface_hub parallel engine.
    """
    print("\n[XM3600 Raw Download]")
    target_dir = "dataset/xm3600"
    os.makedirs(target_dir, exist_ok=True)

    # Check if files already exist to skip downloading
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 5:
        print(f"  Dataset already exists in '{target_dir}'. Skipping.")
        return

    try:
        print("  Downloading raw dataset archive via HF Hub snapshot engine...")

        # Download everything in the repository (images + structural splits)
        snapshot_download(
            repo_id="dinhanhx/crossmodal-3600",
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            # Ignore caching configuration text files you don't need
            ignore_patterns=[".git*", "README.md", "*.py"]
        )

        print(f"  ✅ Success! XM3600 raw dataset downloaded cleanly into: '{target_dir}'")
        print("  Contents:", os.listdir(target_dir))

    except Exception as e:
        print(f"  ❌ Error downloading dataset: {e}")

def download_cifar10_dataset():
    """
    Downloads the CIFAR-10 test split directly to the target folder using
    torchvision's built-in dataset downloader.
    """
    print("\n[CIFAR-10 Raw Download]")
    target_dir = "dataset/cifar10"
    os.makedirs(target_dir, exist_ok=True)

    # Check if files already exist to skip downloading
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"  Dataset already exists in '{target_dir}'. Skipping.")
        return

    try:
        print("  Downloading CIFAR-10 test split via torchvision...")

        # download=True fetches and extracts cifar-10-python.tar.gz automatically
        torchvision.datasets.CIFAR10(root=target_dir, train=False, download=True)

        print(f"  ✅ Success! CIFAR-10 test split downloaded cleanly into: '{target_dir}'")
        print("  Contents:", os.listdir(target_dir))

    except Exception as e:
        print(f"  ❌ Error downloading dataset: {e}")

def download_cifar100_dataset():
    """
    Downloads the CIFAR-100 test split directly to the target folder using
    torchvision's built-in dataset downloader.
    """
    print("\n[CIFAR-100 Raw Download]")
    target_dir = "dataset/cifar100"
    os.makedirs(target_dir, exist_ok=True)

    # Check if files already exist to skip downloading
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"  Dataset already exists in '{target_dir}'. Skipping.")
        return

    try:
        print("  Downloading CIFAR-100 test split via torchvision...")

        # download=True fetches and extracts cifar-10-python.tar.gz automatically
        torchvision.datasets.CIFAR100(root=target_dir, train=False, download=True)

        print(f"  ✅ Success! CIFAR-100 test split downloaded cleanly into: '{target_dir}'")
        print("  Contents:", os.listdir(target_dir))

    except Exception as e:
        print(f"  ❌ Error downloading dataset: {e}")

def download_cvqa_dataset():
    """
    Downloads the CVQA dataset split directly to the target folder using
    Hugging Face's datasets downloader.
    """
    print("\n[CVQA Raw Download]")
    target_dir = "dataset/cvqa"
    os.makedirs(target_dir, exist_ok=True)

    # Check if files already exist to skip downloading
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        print(f"  Dataset already exists in '{target_dir}'. Skipping.")
        return

    try:
        print("  Downloading CVQA dataset via Hugging Face datasets...")

        # Downloads the dataset and caches/saves it to the specified directory
        # You can adjust the split (e.g., 'train', 'test', 'validation') as needed
        REPO = "afaji/cvqa"
        dataset = load_dataset(REPO, split="test", cache_dir=target_dir)

        print(f"  ✅ Success! CVQA dataset downloaded cleanly into: '{target_dir}'")
        print("  Contents:", os.listdir(target_dir))
        return dataset

    except Exception as e:
        print(f"  ❌ Error downloading dataset: {e}")

# --- Configuration & Execution Setup ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download datasets for image retrieval."
    )

    # Dataset choices configuration
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="all",
        choices=["flickr8k", "flickr30k","xtd10","xm3600","cifar10","cifar100","cvqa","all"],
        help="Specify which dataset to download (default: 'all').",
    )

    # Thread configurations
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=8,
        help="Number of threads to use for parallel download (Flickr8k only, default: 8).",
    )

    args = parser.parse_args()

    # Route execution based on command line choices
    if args.dataset in ["flickr8k", "all"]:
        print("=== Starting Flickr8k Download ===")
        download_karpathy_splits()
        download_flickr8k_dataset(num_threads=args.threads)

    if args.dataset in ["flickr30k", "all"]:
        print("\n=== Starting Flickr30k Download ===")
        download_karpathy_splits()
        download_flickr30k_dataset()

    if args.dataset in ["xtd10", "all"]:
        print("\n=== Starting XTD10 Download ===")
        download_xtd10_dataset()

    if args.dataset in ["xm3600", "all"]:
        print("\n=== Starting XTD10 Download ===")
        download_xm3600_dataset()

    if args.dataset in ["cifar10", "all"]:
        print("\n=== Starting CIFAR10 Download ===")
        download_cifar10_dataset()

    if args.dataset in ["cifar100", "all"]:
        print("\n=== Starting CIFAR100 Download ===")
        download_cifar100_dataset()

    if args.dataset in ["cvqa"]:
        print("\n=== Starting CVQA Download ===")
        download_cvqa_dataset()