

import pandas as pd

META_FILES_PATH = "annotations/controlled-conditions/src-files/originals/"

codes_to_ignore = ["91.B"]

def get_insect_codes_from_paper_conditions():
    df = pd.read_csv(META_FILES_PATH + "Insect_position_date.csv")
    df = df[df['comment.insect'].isnull()]
    insect_codes = set()
    for _, row in df.iterrows():
        if row["background"] == "paper":
            code = row["insect.code"]
            if code in codes_to_ignore: continue
            else: code = int(code)
            insect_codes.add(code)

    return sorted(list(insect_codes))

def get_code_to_insect_dict():
    df = pd.read_csv(META_FILES_PATH + "Insect_code_list.csv")
    code_to_insect = {}
    for _, row in df.iterrows():
        code_to_insect[row["insect.code"]] = row["scientificName"]
    return code_to_insect

def get_codes_with_comment_dict():
    df = pd.read_csv(META_FILES_PATH + "Insect_code_list.csv")
    df_with_comments = df[df['Comments'].notnull()]
    code_to_comment = {}
    for _, row in df_with_comments.iterrows():
        code_to_comment[row["insect.code"]] = {"name": row["scientificName"], "comment": row["Comments"]}
    return code_to_comment



name_mappings = { # corrections for typos and redundancies that weren't caught by stanadardizing the name
    "amara ritida": "amara nitida",
    "carabid larva": "carabid larvae",
    "forficula auricularia male": "forficula auricularia",
    "nebria brevicolis": "nebria brevicollis",
    "neottiura binaculata": "neottiura bimaculata",
    "notophilus biguttatus": "notiophilus biguttatus",
    "pterostichus strenus": "pterostichus strenuus"
    }

names_to_group = {
    "achenium depressum": "staphylinidae",
    "acupalpus meridianus": "carabidae",
    "agonum muelleri": "carabidae",
    "aleochara": "staphylinidae",
    "aleocharinae": "staphylinidae",
    "amara": "carabidae",
    "amara aenea": "carabidae",
    "amara convexior": "carabidae",
    "amara ovata": "carabidae",
    "amara nitida": "carabidae",
    "amara similata": "carabidae",
    "anchomenus dorsalis": "carabidae",
    "anotylus": "staphylinidae",
    "armadillidium": "isopoda",
    "asaphidion flavipes": "carabidae",
    "bembidion lampros": "carabidae",
    "bembidion lunulatum": "carabidae",
    "bembidion obtrissum": "carabidae",
    "bembidion properans": "carabidae",
    "brachinus crepitans": "carabidae",
    "brassicogethes": "nitidulidae",
    "nebria brevicolis": "carabidae",
    "notophilus biguttatus": "carabidae",
    "philonthus cognatus": "staphylinidae",
    "cantharis flavilabris": "cantharidae",
    "cantharis lateralis": "cantharidae",
    "cantharis rustica": "cantharidae",
    "chrysoperla carnea": "chrysopidae",
    "coccinella septempunctata": "coccinellidae",
    "cylindroiulus": "myriapoda",
    "drassyllus pusillus": "arachnida",
    "forficula auricularia": "dermaptera",
    "harpalus affinis": "carabidae",
    "harpalus rufipes": "carabidae",
    "hoverfly larva": "diptera larvae",
    "ischnosoma": "staphylinidae",
    "larnycetes emarginatus": "myriapoda",
    "lasius": "formicidae",
    "lepthyphantes": "arachnida",
    "linyphiidae": "arachnida",
    "loricera pilicornis": "carabidae",
    "loricera pilicornis larva": "carabid larvae",
    "lycosidae": "arachnida",
    "megalepthyphantes": "arachnida",
    "micaria pulicaria": "arachnida",
    "microlestes minutulus": "carabidae",
    "myrmica": "formicidae",
    "nebria brevicollis": "carabidae",
    "nebria salina": "carabidae",
    "neottiura bimaculata": "arachnida",
    "notiophilus biguttatus": "carabidae",
    "ocypus olens": "staphylinidae",
    "pardosa": "arachnida",
    "phalangiidae": "arachnida",
    "philonthus cognatus": "staphylinidae",
    "philonthus laminatus": "staphylinidae",
    "philoscia muscorum": "isopoda",
    "poecilus cupreus": "carabidae",
    "polydesmus": "myriapoda",
    "psylloides chrysocephala": "chrysomelidae",
    "pterostichus madidus": "carabidae",
    "pterostichus niger": "carabidae",
    "pterostichus nigrita": "carabidae",
    "pterostichus strenuus": "carabidae",
    "staphylinidae larva": "staphylinidae larvae",
    "stomis punicatus": "carabidae",
    "tachinus": "staphylinidae",
    "trechus quadristriatus": "carabidae",
    "trochosa": "arachnida",
    "trochosa ruricola": "arachnida",
    "xantholinus": "staphylinidae",
    "xantholinus linearis": "staphylinidae",
    "xysticus cristatus": "arachnida"
}

categories_to_set_to_unknown = []