import json
import os
import argparse
import google.generativeai as genai
import time


# API_KEY = os.environ.get('GENAI_API_KEY')
# genai.configure(api_key=API_KEY)

parser = argparse.ArgumentParser()                                                         
parser.add_argument('--json_path', type=str, help='path to dataset to overlay backgrounds')
parser.add_argument('--out_path', type=str, help='path to dataset to overlay backgrounds')

args = parser.parse_args()  

json_path = args.json_path

def get_prompt(scene_graph):
    query = f"Convert the following scene graph into English sentences that describe the scene {scene_graph}."
    " Please provide only the scene description based on the scene graph alone. A valid description is: "
    "A is to the left of B. B is to the right of A and C. C is behind A. A is in front"
    " of C. C is to the left of B."
    return query

import re



def clean_and_merge_text(text):
    """
    Cleans a scene description text by:
    1. Removing introductory phrases like "Here's a description..."
    2. Removing sentences containing unwanted words like "description", "sentences", etc.
    3. Combining bullet points into a paragraph.
    4. Merging multiple paragraphs into a single paragraph.

    Args:
        text (str): The input scene description text.

    Returns:
        str: The cleaned and merged text as a single paragraph.
    """

    # 1. Remove introductory phrases (handling variations)
    text = re.sub(
        r"(?i)^here's a description.*?:\s*",  # Matches variations up to a colon
        '',
        text,
        flags=re.MULTILINE
    )

    # 2. Remove sentences containing unwanted words
    UNWANTED_WORDS = [
        r'\bdescription\b',
        r'\bsentences\b',
        r'\bscene graph\b',
        r'\bokay\b', r'\bok\b',
        r"\bhere's\b",
        r'\bdescribed\b',
        r'\bEnglish\b',
        r'\bprovided\b',
        r'\barrangement\b'
    ]
    unwanted_pattern = re.compile(
        r"(^|(?<=[.!?]))[^\n]*?\b(" + '|'.join(UNWANTED_WORDS) + r")\b[^\n]*?[.!?](?=\s|$)",
        flags=re.IGNORECASE
    )
    text = unwanted_pattern.sub('', text)

    # 3. Handle bullet points (assuming "-", "*", or numbered lists as bullets)
    text = re.sub(r"^\s*[\*\-\d]+\.\s*", '', text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[\*\-]\s*", '', text, flags=re.MULTILINE)

    # 4. Merge multiple lines and paragraphs into a single paragraph
    text = re.sub(r"\s*\n\s*", ' ', text)  # Replace all newlines with a space

    # 5. Clean up excessive spaces
    text = re.sub(r"\s+", ' ', text).strip()

    return text

def rename_objects_with_alphabet(scene_map):
    """
    Replaces object names in the scene map with A, B, C, ... based on sorted object keys.

    Args:
        scene_map (dict): The scene graph with object relationships.

    Returns:
        dict: A new scene graph with object names replaced by A, B, C, ...
    """
    # 1. Get sorted object keys
    sorted_keys = sorted(scene_map.keys())

    # 2. Create mapping from original keys to alphabet labels (A, B, C, ...)
    key_to_label = {key: chr(65 + i) for i, key in enumerate(sorted_keys)}

    # 3. Create a new map with replaced keys and values
    renamed_map = {}
    for old_key, relations in scene_map.items():
        new_key = key_to_label[old_key]
        renamed_relations = {
            relation: [key_to_label[obj] for obj in objects]
            for relation, objects in relations.items()
        }
        renamed_map[new_key] = renamed_relations

    return renamed_map


with open(json_path, 'r') as f:
    data = json.load(f)

out_data = {}



model = genai.GenerativeModel(model_name='gemini-1.5-pro')

for scene_path in data:
    out_data[scene_path] = {}
    for view in data[scene_path]:
        scene_graph = data[scene_path][view]
        scene_graph = rename_objects_with_alphabet(scene_graph)
        prompt = get_prompt(scene_graph)
        response = model.generate_content(prompt)
        out = clean_and_merge_text(response.text)
        out_data[scene_path][view] = out
        print(out)
        print('---------')
        # time.sleep(5)


with open(args.out_path, 'w') as f:
    json.dump(out_data, f, indent=4)