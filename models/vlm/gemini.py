import google.generativeai as genai
from models.base_VLM import BaseVLModel
import models.utils as utils
import os
from PIL import Image
import numpy as np

API_KEY = os.environ.get('GENAI_API_KEY')
genai.configure(api_key=API_KEY)

class Gemini(BaseVLModel):
    def __init__(self, args):
        self.model = genai.GenerativeModel(model_name='gemini-2.0-flash')

    def get_prompt(self, cat, other_bboxes=None, other_cats=None, scene_graph=None):
        query = f"Return bounding boxes for object named {cat} in the image in the following format as"
        " a single list. \n [[ymin, xmin, ymax, xmax], object_label, certainty_score] \n"
        if other_cats is not None and len(other_cats) > 0:
            add = 'There are 3 objects in the scene. They are: '
            for ind, c in enumerate(other_cats):
                add += c[0].split('_')[0]
                if ind < len(other_cats) - 1:
                    add += ', '
                else:
                    add += '.'
            query = add + f" {query}"
        if scene_graph is not None:
            query = f"Given the scene description: {scene_graph} {query}"

        return query
    
    def build_input_ids(self, image, query, history):
        inputs = [image, query]
        return inputs
    
    def get_model_input(self, input_by_model):
        
        return input_by_model
    
    def generate_content(self, inputs):
        response = self.model.generate_content(inputs)

        return response
    
    def process_model_output(self, inputs, response):
        result = response.text
        bbox = utils.extract_bounding_box(result)  #### bbox in [ymin, xmin, ymax, xmax]
        if bbox:
            ymin, xmin, ymax, xmax = bbox
            bbox = [xmin, xmax, ymin, ymax]
        # print(response)
        return result, bbox
    
    def process_image(self, image_path, other_bboxes, other_cats, mask=False):
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        if other_cats is not None and len(other_cats) > 0 and mask:
            # import pdb; pdb.set_trace()
            image_np = np.array(image_np)
            image_cat = np.zeros(image_np.shape,dtype=np.bool_)

            for bbox in other_bboxes:
                seg = bbox[0]
                seg = np.array(seg).astype(np.bool_)
                seg = np.stack([seg,seg,seg],axis=2)
                image_cat[seg] = True
            image_np = image_np * image_cat
        image = Image.fromarray(np.array(image_np))
        return image