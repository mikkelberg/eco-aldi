import json

def save_json_to_file(json_obj, path):
    with open(path, "w") as f:
        json.dump(json_obj, f, indent=4)

def load_json_from_file(path):
    with open(path, "r") as f:
        json_obj = json.load(f)
    return json_obj