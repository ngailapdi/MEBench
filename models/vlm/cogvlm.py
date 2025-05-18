import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from models.base_VLM import BaseVLModel
import models.utils as utils
from PIL import Image
import cv2
import numpy as np

class CogVLM(BaseVLModel):
    def __init__(self, args):
        MODEL_PATH = args.from_pretrained
        TOKENIZER_PATH = args.local_tokenizer
        if args.torch_dtype == 'bf16':
            self.torch_type = torch.bfloat16
        else:
            self.torch_type = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=self.torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=True,
            trust_remote_code=True
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        self.get_model_kwargs()

    def get_prompt(self, cat, other_bboxes=None, other_cats=None, scene_graph=None):
        query = "Where is the %s? Let's think step by step"%(cat)

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
        input_by_model = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=history, images=[image])

        return input_by_model

    def generate_content(self, inputs):
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_kwargs)
        return outputs

    def process_model_output(self, inputs, outputs):
        outputs = outputs[:, inputs['input_ids'].shape[1]:][0]

        response = self.tokenizer.decode(outputs)
        response = response.split("</s>")[0]
        bbox = utils.extract_bounding_box(response)  #### bbox in [xmin, xmax, ymin, ymax]

        return response, bbox
    
    def get_model_input(self, input_by_model):
        input_ids = torch.tensor(input_by_model['input_ids']).unsqueeze(0).to(self.device)
        token_type_ids = torch.tensor(input_by_model['token_type_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(input_by_model['attention_mask']).unsqueeze(0).to(self.device)
        images = [[img.to(self.device).to(self.torch_type) for img in input_by_model['images']]]
        
        inputs = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'images': images,
        }
        return inputs
    
    def get_model_kwargs(self):
        self.gen_kwargs = {"max_length": 2048,
                "do_sample": False} # "temperature": 0.9

    def process_image(self, image_path, other_bboxes=None, other_cats=None, mask=False):
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
        import pdb; pdb.set_trace()
        return image