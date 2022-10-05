import glob
import json
import os


def get_data_list(paths, folders):
    data = []
    for path in paths:
        for i in range(len(folders)):
            for img_path in glob.glob(path + '/' + folders[i] + '/*'):
                data.append({
                    "path": f"{img_path}",
                    "label": i
                })
    return data


data_list = {
    "train": get_data_list(["data/raw/training"], ["non_food", "food"]),
    "val": get_data_list(["data/raw/validation"], ["non_food", "food"]),
    "eval": get_data_list(["data/raw/evaluation"], ["non_food", "food"])
}

os.makedirs("data/prepared", exist_ok=True)

with open("data/prepared/data_list.json", "w") as file:
    json.dump(data_list, file)
