import json
import os
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_image(image_info):
    url, save_path = image_info
    
    if os.path.exists(save_path):
        return # Skip if already exists
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download {url}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download COCO images for CapRL")
    parser.add_argument("--dataset_path", type=str, default="caprl_mcq_dataset_final.jsonl", help="Path to the dataset JSONL file")
    parser.add_argument("--output_dir", type=str, default="data/images", help="Directory to save images")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of threads for downloading")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Base URL for COCO 2017 Train images
    base_url = "http://images.cocodataset.org/train2017/"

    unique_images = set()
    
    print(f"Reading dataset from {args.dataset_path}...")
    try:
        with open(args.dataset_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # Extract image path/id. 
                    # Format in file: "image_id": "coco/train2017/000000012209.jpg"
                    # We need just the filename: 000000012209.jpg
                    image_path_raw = entry.get("image_id", "")
                    if "train2017/" in image_path_raw:
                        filename = image_path_raw.split("train2017/")[-1]
                        unique_images.add(filename)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: File {args.dataset_path} not found.")
        return

    print(f"Found {len(unique_images)} unique images to download.")

    download_tasks = []
    for filename in unique_images:
        url = base_url + filename
        save_path = os.path.join(args.output_dir, filename)
        download_tasks.append((url, save_path))

    print("Starting download...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(tqdm(executor.map(download_image, download_tasks), total=len(download_tasks)))

    print("Download complete.")

if __name__ == "__main__":
    main()
