from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
from models.base_VLM import BaseVLModel
import models.utils as utils
import torch
import cv2

class Sa2VA(BaseVLModel):
    def __init__(self, args):
        MODEL_PATH = args.from_pretrained
        if args.torch_dtype == 'bf16':
            self.torch_type = torch.bfloat16
        else:
            self.torch_type = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=self.torch_type,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )


        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    def get_prompt(self, cat, other_bboxes=None, other_cats=None, scene_graph=None):
        # query = "Where is the %s? Let's think step by step."%(cat)
        query = "Can you segment the %s?"%(cat)


        if other_cats is not None and len(other_cats) > 0:
            add = f'There are 3 objects in the scene. They are: '
            for ind, c in enumerate(other_cats):
                add += c[0].split('_')[0]
                if ind < len(other_cats) - 1:
                    add += ', '
                else:
                    add += '.'
            query = '<image>' + add + f" {query}"
        elif scene_graph is None:
            query = '<image>' + f"{query}"
        else:
            query = '<image>' + f"Given the scene description: {scene_graph} {query}"

        return query
    
    def build_input_ids(self, image, query, history):
        input_by_model = {'images': [image], 'query': query, 'history': history}
        return input_by_model

    def get_model_input(self, input_by_model):
        images = [img for img in input_by_model['images']]
        query = input_by_model['query']
        history = input_by_model['history']
        return {'images': images, 'query': query, 'history':history}
    
    def generate_content(self, inputs):
        with torch.no_grad():
            result = self.model.predict_forward(
                image=inputs['images'][0],
                text=inputs['query'],
                tokenizer=self.tokenizer,

            )
        return result
    
    def process_model_output(self, inputs, outputs):
        response = outputs['prediction']
        mask = outputs['prediction_masks']
        return response, mask
    
    def process_image(self, image_path, other_bboxes=None, other_cats=None, mask=False):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        if other_cats is not None and len(other_cats) > 0 and mask:
            # import pdb; pdb.set_trace()
            image = np.array(image)
            image_cat = np.zeros(image.shape,dtype=np.bool_)

            for bbox in other_bboxes:
                seg = bbox[0]
                seg = np.array(seg).astype(np.bool_)
                seg = np.stack([seg,seg,seg],axis=2)
                image_cat[seg] = True
            image = image * image_cat
        image = Image.fromarray(image)
        return image