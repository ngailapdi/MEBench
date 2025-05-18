import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer
from models.base_VLM import BaseVLModel
import models.utils as utils
from xtuner.registry import BUILDER
import os

from xtuner.configs import cfgs_name_path
from mmengine.config import Config, DictAction
from mmengine.fileio import PetrelBackend, get_file_backend
from xtuner.model.utils import guess_load_checkpoint
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)
from xtuner.tools.utils import get_stop_criteria
from transformers.generation.streamers import TextStreamer

from xtuner.dataset.utils import expand2square, load_image
import numpy as np
from PIL import Image



class OMG_LLaVA(BaseVLModel):
    def __init__(self, args):
        cfg = Config.fromfile(args.config)
        model_name = cfg.model.type if isinstance(cfg.model.type,
                                              str) else cfg.model.type.__name__
        if 'LLaVAModel' or 'OMG' in model_name:
            cfg.model.pretrained_pth = None

        self.model = BUILDER.build(cfg.model)
        backend = get_file_backend(args.pth_model)
        if isinstance(backend, PetrelBackend):
            from xtuner.utils.fileio import patch_fileio
            with patch_fileio():
                state_dict = guess_load_checkpoint(args.pth_model)
        else:
            state_dict = guess_load_checkpoint(args.pth_model)
        self.model.load_state_dict(state_dict, strict=False)
        self.bits = args.bits
        self.offload_folder = args.offload_folder

        TORCH_DTYPE_MAP = dict(
                fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')
        self.torch_type = TORCH_DTYPE_MAP[args.torch_dtype]
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.repetition_penalty = args.repetition_penalty
        self.llm = self.model.llm
        self.tokenizer = self.model.tokenizer
        self.stop_words = args.stop_words
        
        self.prompt_template = args.prompt_template
        self.bot_name = args.bot_name
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        self.llm.eval()
        self.visual_encoder = self.model.visual_encoder
        self.projector = self.model.projector
        self.projector_text2vision = self.model.projector_text2vision
        self.get_model_kwargs()
        self.stop_criteria = []
        self.no_streamer = args.no_streamer
        self.image_processor = cfg.image_processor
        self.image_processor_type = self.image_processor['type']
        del self.image_processor['type']
        self.image_processor = self.image_processor_type(**self.image_processor)

        if self.prompt_template:

            self.template = PROMPT_TEMPLATE[self.prompt_template]

            self.stop_words += self.template.get('STOP_WORDS', [])
            self.stop_criteria = get_stop_criteria(tokenizer=self.tokenizer, stop_words=self.stop_words)

    def get_model_kwargs(self):
        quantization_config = None
        load_in_8bit = False
        if self.bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4')
        elif self.bits == 8:
            load_in_8bit = True
        self.gen_kwargs = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
        )

    def get_prompt(self, cat, other_bboxes=None, other_cats=None, scene_graph=None):
        query = "Please provide segmentation mask for the %s?"%(cat)
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

        text = DEFAULT_IMAGE_TOKEN + '\n' + query

        if self.prompt_template:
            prompt_text = ''


            prompt_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
 
        else:
            prompt_text = text

        query = prompt_text

        return query
    
    def build_input_ids(self, image, query, history):
        chunk_encode = []
        for idx, chunk in enumerate(query.split(DEFAULT_IMAGE_TOKEN)):
            if idx == 0:
                cur_encode = self.tokenizer.encode(chunk)
            else:
                cur_encode = self.tokenizer.encode(
                    chunk, add_special_tokens=False)
            chunk_encode.append(cur_encode)
        assert len(chunk_encode) == 2
        ids = []
        for idx, cur_chunk_encode in enumerate(chunk_encode):
            ids.extend(cur_chunk_encode)
            if idx != len(chunk_encode) - 1:
                ids.append(IMAGE_TOKEN_INDEX)
        ids = torch.tensor(ids).cuda().unsqueeze(0)
        input_by_model = {'ids': ids, 'pixel_values': image}
        return input_by_model

    def get_model_input(self, input_by_model):
        inputs = prepare_inputs_labels_for_multimodal(
            llm=self.llm, input_ids=input_by_model['ids'], pixel_values=input_by_model['pixel_values'])
        return inputs
    
    def generate_content(self, inputs):
        if self.no_streamer:
            streamer = None
        else:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        generate_output = self.llm.generate(
            **inputs,
            generation_config=self.gen_kwargs,
            streamer=streamer,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        return generate_output

    def process_model_output(self, inputs, outputs):
        def get_seg_hidden_states(hidden_states, output_ids, seg_id):
            seg_mask = output_ids == seg_id
            n_out = len(seg_mask)
            return hidden_states[-n_out:][seg_mask]
        hidden_states = outputs.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, outputs.sequences[0][:-1],
            # last_hidden_states, generate_output.sequences[0],
            seg_id=self.model.seg_token_idx
        )
        if len(seg_hidden_states) != 0:
            seg_hidden_states = self.projector_text2vision(seg_hidden_states)
            batch_idxs = torch.zeros((seg_hidden_states.shape[0], ),
                                        dtype=torch.int64).to(seg_hidden_states.device)
            pred_masks_list = self.model.visual_encoder.forward_llm_seg(seg_hidden_states, batch_idxs)
            mask = pred_masks_list[-1].sigmoid() > 0.5
            mask = mask.cpu().numpy()
        else:
            mask = None


        output_text = self.tokenizer.decode(outputs.sequences[0])
        end = '' if output_text[-1] == '\n' else '\n'
        return output_text, mask

    def process_image(self, image_path, other_bboxes=None, other_cats=None, mask=False):
        image = load_image(image_path)
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
        image = expand2square(
            image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        image = self.image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda().unsqueeze(0).to(self.visual_encoder.dtype)
        visual_outputs = self.visual_encoder(image, output_hidden_states=True)
        pixel_values = self.projector(visual_outputs)
        return pixel_values