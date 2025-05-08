import json
import os
import argparse

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils 

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Compute and add (or recompute) the area-field in the annotations.")
    parser.add_argument("dataset_name", help="E.g. pitfall-cameras (matches the directory)")

    # python scripts/gen_coco.py pitfall-cameras ../../../mnt/data0/martez/pitfall-cameras/images/

    # Parse the arguments
    args = parser.parse_args()

    src_dir = "annotations/" + args.dataset_name + "/"

    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]

    for file in files: 

        coco = utils.load_json_from_file(file)
        print("\n" + file + ":")

        total_anns = len(coco["annotations"])

        for i, ann in enumerate(coco["annotations"], start=1):
            if i % 500 == 0 or i == total_anns: print(f"-- Processed {i} out of {total_anns} annotations.")
            ann["area"] = ann["bbox"][2] * ann["bbox"][3]
            ann["segmentation"] = []
            ann["iscrowd"] = 0
        
        utils.save_json_to_file(coco,file)
    print("\n\nDone!")

if __name__ == "__main__":
    main()