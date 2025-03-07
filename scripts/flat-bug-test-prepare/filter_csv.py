from pathlib import Path
import pandas as pd
from pandas import DataFrame, Series

pd.options.mode.copy_on_write = True # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

root = Path("data-annotations/controlled-conditions/annotations")
csv_file = root.joinpath("Insect_position_date.csv")

def format_date_column(df: DataFrame):
    """Format date column in dataframe series: ddmmyy"""
    dates = df['date']
    for i, date in enumerate(dates):
        date = date.split("/")
        date[2] = date[2][2:4]
        dates[i] = "".join(date)
    
    print(dates)

def filter_meta_csv(df: DataFrame):
    """We only want paper background and in the range of 15/05-07/06 2023"""
    format_date_column(df)

    paper_rows = df[df['background'] == 'paper']
    
    # TODO Remove extra dates

# Format date column: ddmmyy
# Use date column + camera column to load 'Annotations/<date> <camera>.json' 
# Load via json module --> Filter values under key "_via_img_metadata" such that only annotated entries remain
# Convert filtered json values to list and use an rng to index 10 random elements --> Locate images in flat-bug dataset --> Save their paths 
# Write paths to a .txt file and load via 7z to extract only selected images from archive 
# Run flat-bug


if __name__ == '__main__':
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    
    # Remove entries that are too new from dataframe
    #df['date']

    # Filter annotations-paper accordingly

    #print(df)
    