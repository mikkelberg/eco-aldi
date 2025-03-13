

import pandas as pd

META_FILES_PATH = "annotations/controlled-conditions/info/meta-files/"
def get_insect_codes_from_paper_conditions():
    df = pd.read_csv(META_FILES_PATH + "/Insect_position_date.csv")
    insect_codes = set()
    for _, row in df.iterrows():
        if row["background"] == "paper":
            code = row["insect.code"]

            if code in codes_to_ignore:
                continue
            else:
                code = int(code)
            
            insect_codes.add(code)
    return sorted(list(insect_codes))

codes_to_ignore = ["91.B"]


name_mappings = { # corrections for typos and redundancies that weren't caught by stanadardizing the name
    }

names_to_group = {
}

categories_to_set_to_unknown = []