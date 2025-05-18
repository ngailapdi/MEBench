import os
import json
import numpy as np

old_json_path = '../jsons/toys_cate_split.json'
new_json_path = '../jsons/toys_cate_split_new.json'
novel_objs_path = '../jsons/geoshape20_dict.json'

with open(old_json_path, 'r') as f:
    data = json.load(f)

train_shots = data['train_shot']
test_shots = data['test_shot']
base_1 = data['base_1']
base_2 = data['base_2']

base_1.update(train_shots)
base_2.update(test_shots)
base_1.update(data['others'])
new_data = {'base_1': base_1, 'base_2': base_2}

with open(novel_objs_path, 'r') as f:
    data = json.load(f)

novel_obj_data = {d: [d] for d in data}



new_data = {'base_1': base_1, 'base_2': base_2, 'train_shot': novel_obj_data, 'test_shot': novel_obj_data}


with open(new_json_path, 'w') as f:
    json.dump(new_data, f, indent=4)