import json
import os
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils 

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Gen list of unused images acc. to the coco json files.")
    parser.add_argument("dataset_name", help="E.g. pitfall-cameras (matches the directory)")
    parser.add_argument("images_dir", help="Path to images directory.")

    # python scripts/gen_coco.py pitfall-cameras ../../../mnt/data0/martez/pitfall-cameras/images/

    # Parse the arguments
    args = parser.parse_args()

    src_dir = "annotations/" + args.dataset_name + "/"
    images_dir = args.images_dir
    dest_path = src_dir + "info/unused-images.txt"
    
    included_images = set()
    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
    for index, filename in enumerate(files, start=1): 
        print(filename)
        coco = utils.load_json_from_file(filename)
        for img in coco["images"]: 
            included_images.add(img["file_name"])


    #stop = 0 
    unused_images = []
    total_images = 0
    for root, _, files in os.walk(images_dir):
        print(f"Processing folder: {root}.")
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue  # Ensure it's an image file
            image_path = os.path.join(root, file)
            relative_path = os.path.relpath(image_path, images_dir)
            if relative_path not in included_images: unused_images.append(relative_path)
            total_images += 1
        #stop += 1
        #if stop == 5: break
    
    with open(dest_path, 'w') as f:
        for path in unused_images:
            f.write(f"{path}\n")

    print(f"\nDone! There were {total_images} in total, and {len(list(included_images))} were included. {len(unused_images)} were listed to remove.")
if __name__ == "__main__":
    main()