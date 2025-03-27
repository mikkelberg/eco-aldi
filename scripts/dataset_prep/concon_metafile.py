import os
import pandas as pd
from pandas import DataFrame
import sys

from extract_coco_categories import normalised_name_to_concon_code, normalise_category_name
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.utils as utils
import utils.controlled_conditions_utils as concon

META_FILES_PATH = "annotations/controlled-conditions/src-files/"

codes_to_ignore = ["91.B"]

def format_date_column(df: DataFrame):
    """Format date column in dataframe series: ddmmyy"""
    df['date'] = df['date'].apply(format_date)
    return df

def format_date(date):
    date = date.split("/")
    date[2] = date[2][2:4]
    date = "".join(date)
    return date

df = pd.read_csv(META_FILES_PATH + "Insect_position_date.csv")
df.drop(['camera.type', 'position', 'camera.check', 'recorder'], axis=1, inplace=True)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df.drop(df[df['background'] != "paper"].index, inplace=True)
df.drop(df[df['comment.insect'].notnull()].index, inplace=True)
df.drop(['background', 'comment.insect'], axis=1, inplace=True)

format_date_column(df)

code_to_insect = concon.get_code_to_insect_dict()
df["insect_name"] = df['insect.code'].map(code_to_insect)
df.drop(['insect.code'], axis=1, inplace=True)
df.drop(df[df['insect_name'].isnull()].index, inplace=True)
df['insect_name'] = df['insect_name'].apply(normalise_category_name)
df['og_insect_name'] = df['insect_name']

mother = utils.load_json_from_file("annotations/categories.json")
cats_to_remove = mother["remove"].keys()
name_to_official_category = {}
for off_cat, info in mother["categories"].items():
    for n in info["contains"]:
        name_to_official_category[n] = off_cat

df['insect_name'] = df['insect_name'].map(name_to_official_category)
df.drop(df[df['insect_name'].isnull()].index, inplace=True)

print(df)

df.to_csv('annotations/controlled-conditions/info/meta.csv', index=False)