import os
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from models.base_VLM import BaseVLModel
sys.path.insert(1, '/home/ant/develop/LISA')
sys.path.insert(1, '/home/ant/develop/LISA/model/')


from model.LISA import LISAForCausalLM
import model.llava.model
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

class LISA(BaseVLModel):
    def __init__(self, args):
        model_path = args.from_pretrained
        if args.torch_dtype == 'bf16':
            self.torch_type = torch.bfloat16
        else:
            self.torch_type = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=None,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        args.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

        self.get_model_kwargs()

        self.model = LISAForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **self.kwargs
        )
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        self.vision_tower = self.model.get_model().get_vision_tower()
        self.vision_tower.to(dtype=self.torch_type)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = ResizeLongestSide(args.glamm_image_size)

    def get_model_kwargs(self):
        self.kwargs = {
            'torch_dtype': self.torch_type,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        }

    def preprocess(
        self, x,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024,
    ) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - pixel_mean) / pixel_std
        # Pad
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def process_image(self, image_path, other_bboxes=None, other_cats=None, mask=False):
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
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
            # import pdb; pdb.set_trace()
        original_size_list = [image_np.shape[:2]]
        image_clip = (
            self.clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        image_clip = image_clip.to(self.torch_type)


        image = self.transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        image = image.to(self.torch_type)
        return {'image': image, 'image_clip': image_clip, 'resize_list': resize_list, \
                'original_size_list': original_size_list}
    
    def build_input_ids(self, image, query, history):
        input_ids = tokenizer_image_token(query, self.tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()
        return {'input_ids': input_ids, **image}
    
    def get_model_input(self, input_by_model):
        return input_by_model

    def generate_content(self, inputs):
        with torch.no_grad():
            output_ids, pred_masks = self.model.evaluate(
                inputs['image_clip'],
                inputs['image'],
                inputs['input_ids'],
                inputs['resize_list'],
                inputs['original_size_list'],
                max_new_tokens=512,
                tokenizer=self.tokenizer,
            )
        return {'output_ids': output_ids, 'pred_masks': pred_masks}
    
    def process_model_output(self, inputs, outputs):
        output_ids = outputs['output_ids']
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        text_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        response = text_output.replace("\n", "").replace("  ", " ")

        bbox = [outputs['pred_masks'][0].detach().cpu().numpy()]

        return response, bbox

    def get_prompt(self, cat, other_bboxes=None, other_cats=None, scene_graph=None):
        query = "Can you segment the %s?"%(cat)

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
        
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + query
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        query = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            

        return query