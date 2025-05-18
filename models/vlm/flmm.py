import torch
import numpy as np
from mmengine.config import Config
from xtuner.registry import BUILDER
from PIL import Image
from xtuner.model.utils import guess_load_checkpoint
from models.base_VLM import BaseVLModel

import sys
for mod in list(sys.modules):
    if mod.startswith("segment_anything"):
        del sys.modules[mod]

import sys
sys.path.insert(0, '/home/ant/develop/F-LMM')

import spacy
nlp = spacy.load("en_core_web_sm")


class FLMM(BaseVLModel):
    def __init__(self, args):
        self.cfg = Config.fromfile(args.config)
        self.prompt_template = self.cfg.prompt_template
        self.tokenizer = self.cfg.tokenizer
        image_processor = self.cfg.image_processor
        prompt = self.cfg.get('prompt', None)

        self.model = BUILDER.build(self.cfg.model)
        state_dict = guess_load_checkpoint(args.pth_model)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model._prepare_for_generation(image_processor=image_processor,
                                    prompt_template=self.prompt_template,
                                    max_thought_tokens=16,
                                    max_new_tokens=512,
                                    lmm_name=self.cfg.lmm_name,
                                    additional_prompt='')
        self.model = self.model.cuda().eval()

    def get_prompt(self, cat, other_bboxes=None, other_cats=None, scene_graph=None):
        query = "Where is the %s?"%(cat)

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

        return {'query': query, 'cat': cat}

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
        return image
    
    def build_input_ids(self, image, query, history):
        return {'image': image, **query}
    
    def get_model_input(self, input_by_model):
        return input_by_model

    def process_noun_chunks(self, noun_chunks):
        new_noun_chunks = []
        for i in range(len(noun_chunks)):
            noun_chunk = noun_chunks[i]
            if 'image' in noun_chunk.lower():
                continue
            if noun_chunk.lower() in ['it', 'this', 'that', 'those', 'these', 'them',
                                    'he', 'she', 'you', 'i', 'they', 'me', 'her',
                                    'him', 'a', 'what', 'which', 'whose', 'who']:
                continue
            keep = True
            for j in range(len(noun_chunks)):  # de-duplicate
                if i != j and noun_chunk in noun_chunks[j]:
                    if len(noun_chunk) < len(noun_chunks[j]) or i > j:
                        keep = False
                        break
            if keep:
                new_noun_chunks.append(noun_chunk)

        return new_noun_chunks
    
    def extract_noun_phrases(self, output_text, cat):
        doc = nlp(output_text)
        noun_chunks = list(set(chunk.text for chunk in doc.noun_chunks))
        if len(noun_chunks) == 0:
            noun_chunks = [output_text]
        last_end = 0
        noun_chunks = self.process_noun_chunks(noun_chunks)
        noun_chunks = sorted(noun_chunks, key=lambda x: output_text.find(x))

        # noun_chunks = [noun_chunk for noun_chunk in noun_chunks
        #                if int(input(f'Ground {noun_chunk}?')) == 1]
        noun_chunks = [noun_chunk for noun_chunk in noun_chunks
                    if cat in noun_chunk]

        positive_ids = []
        phrases = []
        for noun_chunk in noun_chunks:
            obj_start = output_text.find(noun_chunk)
            if obj_start < last_end:
                continue
            obj_end = obj_start + len(noun_chunk)
            last_end = obj_end
            positive_ids.append((obj_start, obj_end))
            phrases.append(noun_chunk)

        return positive_ids, phrases

    def find_interval(self, intervals, idx):
        for interval_id, (start_id, end_id) in enumerate(intervals):
            if (idx >= start_id) and (idx < end_id):
                return interval_id
        return len(intervals)
    
    def generate_content(self, inputs):
        with torch.no_grad():
            output = self.model.answer(inputs['image'], inputs['query'])
            output_ids = output.pop('output_ids').cpu()

        return {'output': output, 'output_ids': output_ids}
    
    def process_model_output(self, inputs, outputs):
        output = outputs['output']
        output_text = output.pop('output_text')
        encoded = self.model.tokenizer(output_text, add_special_tokens=False, return_tensors='pt')
        offsets = encoded.encodings[0].offsets
        str_places, phrases = self.extract_noun_phrases(output_text, inputs['cat'])
        positive_ids = []

        for start_id, end_id in str_places:
            start_token_place = self.find_interval(offsets, start_id)
            end_token_place = max(start_token_place+1, self.find_interval(offsets, end_id))
            positive_ids.append((start_token_place, end_token_place))
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            try:
                pred_masks, sam_pred_masks = self.model.ground(image=inputs['image'], positive_ids=[positive_ids[0]], **output)
            except:
                return output_text, None
        pred_masks = [sam_pred_masks.cpu().numpy() > 0]

        return output_text, pred_masks