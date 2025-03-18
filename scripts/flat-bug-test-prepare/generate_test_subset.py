import json
import os
from pathlib import Path
import pandas as pd
from pandas import DataFrame, Series

pd.options.mode.copy_on_write = True # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

ROOT = Path("data-annotations").joinpath("controlled-conditions", "annotations")
CSV_FILE = ROOT.joinpath("meta-files", "Insect_position_date.csv")
ANNOTATIONS = ROOT.joinpath("Annotations")

IMAGE_FOLDER_DATE_MAP = {"040523":"040523", "150523":"150523", "160523":"160523", "170523":"170522", "050623":"0506-120623", "070623":"0506-120623"}

def format_date_column(df: DataFrame):
    """Format date column in dataframe series: ddmmyy"""
    df['date'] = df['date'].apply(format_date)
    return df

def format_date(date):
    date = date.split("/")
    date[2] = date[2][2:4]
    date = "".join(date)
    return date

def filter_meta_csv(df: DataFrame):
    """We only want paper background and in the range of 15/05-07/06 2023"""

    df = df[df['background'] == 'paper']
    return df

def find_paper_annotations(annotations, dates_cameras):
    paper_annotations = []
    for date, camera in dates_cameras: 
        basename = date + " " + camera + ".json"
        for target in os.listdir(annotations):
            if basename == target:
                paper_annotations.append(target)
    return paper_annotations

def find_images_with_bb(annotation_file):
    with open(annotation_file) as f: 
        file = json.load(f)
        images = file["_via_img_metadata"].values()
        images_with_bb = []
        for image in images: 
            if image['regions']:
                images_with_bb.append(image["filename"])
        
        return images_with_bb

def write_target_subset(test_images, output):
    with open(ROOT.joinpath(output), "w") as f:
        f.write("\n".join(str(image) for image in test_images))

def collect_target_subset(dictionary):
    # Collect image paths to be loaded by 7z, save to a .txt file 
    no_to_sample = 18 # found via printing the number of values for each entry in dictionary. Let the lowest number decide.
    test_images = []
    for annotation_filename in dictionary: 
        annotated_images = list(dictionary[annotation_filename])
        sample_distance = int(len(annotated_images)/no_to_sample) # Round down 
        for index in range(0, len(annotated_images), sample_distance):
            date_camera = annotation_filename.split(" ")
            date = date_camera[0]
            camera = os.path.splitext(date_camera[1])[0]
            test_image = Path(IMAGE_FOLDER_DATE_MAP[date]).joinpath(camera, annotated_images[index])
            test_images.append(test_image)
    return test_images

def generate_test_subset():
    df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
    
    df = filter_meta_csv(df)
    df = format_date_column(df)
    dates_cameras = list(zip(df.date, df.camera))
    
    paper_annotation_filenames = find_paper_annotations(ANNOTATIONS, dates_cameras)
    dictionary = dict()
    for annotation_filename in paper_annotation_filenames:
        dictionary[annotation_filename] = find_images_with_bb(ANNOTATIONS.joinpath(annotation_filename))

    test_images = collect_target_subset( dictionary)
    
    write_target_subset(test_images=test_images, output="flat-bug-test-images.txt")

if __name__ == '__main__':
    generate_test_subset()