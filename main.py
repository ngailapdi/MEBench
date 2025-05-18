import argparse
import torch
import os

print(os.getcwd())
import sys
print(sys.path)
sys.path.append(os.getcwd())

from PIL import Image
from data.basic_dataset import BasicDataset
from data import data_utils

import numpy as np
import json
from tqdm import tqdm
import argparse
import cv2
import time
from torchvision import transforms
import torch.nn.functional as F
sys.path.insert(0, '/home/ant/develop/OMG-Seg/omg_llava')




import random

from models.vlm import get_model

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument('--config', default='', help='config file name or path.')
    parser.add_argument('--pth_model', default='', help='pth model file')

    parser.add_argument('--image', default=None, help='image')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--prompt-template',
        default="internlm2_chat",
        help='Specify a prompt template')
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', default=None, help='Specify the system text')
    system_group.add_argument(
        '--system-template',
        default=None,
        help='Specify a system template')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument(
        '--lagent', action='store_true', help='Whether to use lagent')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible text generation')
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--model_max_length", default=1536, type=int)


    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--glamm_image_size", default=1024, type=int, help="Image size for grounding image encoder")


    parser.add_argument("--from_pretrained", type=str, default="THUDM/cogvlm-grounding-generalist-hf", help='pretrained ckpt')
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
    parser.add_argument("--model", type=str, default="cogvlm", help='model name')
    parser.add_argument("--data_mode", type=str, default="most_visible_object", help='data mode name')
    parser.add_argument("--run", type=str, default="0", help='run number')
    parser.add_argument("--load_seg", type=str, default="", help='run number')
    parser.add_argument("--num_views", type=int, default=25, help='run number')



    args = parser.parse_args()
    return args



def get_bbox_from_seg(seg):
    # Find non-zero (foreground) pixel coordinates
    y_indices, x_indices = np.where(seg > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None  # No object found in the mask

    # Compute bounding box
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    ymin, ymax = np.min(y_indices), np.max(y_indices)

    return [xmin, xmax, ymin, ymax]


def draw_bounding_boxes(image, bounding_boxes_with_labels, resized=(224, 224)):
    label_colors = {}
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = np.array(image)

    for bounding_box in bounding_boxes_with_labels:

        # Normalize the bounding box coordinates
        width, height = image.shape[1], image.shape[0]
        xmin, xmax, ymin, ymax = bounding_box
        # ymin, xmin, ymax, xmax = bounding_box
        # x1 = int(xmin / resized[0] * width)
        # y1 = int(ymin / resized[1] * height)
        # x2 = int(xmax / resized[0] * width)
        # y2 = int(ymax / resized[1] * height)

        # xmin, ymin, xmax, ymax = bounding_box

        # # Scale the bounding box coordinates to match the current image size
        scale_x = width / resized[0]
        scale_y = height / resized[1]
        x1 = int(xmin * scale_x)
        y1 = int(ymin * scale_y)
        x2 = int(xmax * scale_x)
        y2 = int(ymax * scale_y)

        color = np.random.randint(0, 256, (3,)).tolist()

        box_thickness = 2


        # cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        # cv2.putText(image, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

    image = Image.fromarray(image)
    return image

def get_rescaled_bbox(bbox, original_size=(224,224), resized=(224,224)):
    xmin, xmax, ymin, ymax = bbox
    width, height = original_size
    scale_x = width / resized[0]
    scale_y = height / resized[1]
    x1 = int(xmin * scale_x)
    y1 = int(ymin * scale_y)
    x2 = int(xmax * scale_x)
    y2 = int(ymax * scale_y)
    return [x1, x2, y1, y2]

def read_data_dino(img_path, seg_path):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    prep = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    img = Image.open(img_path).convert('RGB')
    try:
        seg = data_utils.read_segmentation(seg_path)
        seg = np.stack([seg, seg, seg], axis=-1)
        seg = Image.fromarray(seg)
        img, seg = data_utils.mask_image(img, seg)
    except:
        seg = []
    img = prep(img)
    if seg != []:
        seg = np.array(seg)[:,:,0]
    return img, seg



def get_dino_base(args, model):
    dataset_base = BasicDataset(mode=args.data_mode, seed=args.seed, num_views=args.num_views, load_seg='gt', custom_root='base')
    dataloader_base = torch.utils.data.DataLoader(
        dataset_base,
        num_workers=0,
        batch_size=1,
        pin_memory=True
    )
    all_features = []
    all_labels = []
    # import pdb; pdb.set_trace()

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader_base), total=len(dataloader_base), desc="Processing Batches"):
            # try:
            img_path = batch['img_path'][0]
            cat = batch['cat'][0]
            obj = batch['obj'][0]
            seg_path = batch['seg_path'][0]
            img, seg = read_data_dino(img_path, seg_path)
            feat = model(img.unsqueeze(0).cuda())
            feat = F.normalize(feat, dim=1)
            all_features.append(feat)
            all_labels.append(cat)
            # except:
            #     import pdb; pdb.set_trace()
    all_features = torch.cat(all_features, dim=0)
    return all_features, all_labels



def main_dino():

    dataset = BasicDataset(mode=args.data_mode, seed=args.seed, num_views=args.num_views, load_seg=args.load_seg)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        pin_memory=True
    )
    all_preds = []
    if args.model == 'dino':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
        model = model.backbone.cuda()

    elif args.model == 'dinov1':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').cuda()
    base_feat, base_label = get_dino_base(args, model)
    sim = torch.zeros((len(dataset), len(base_feat)), dtype=torch.float32)
    all_objs = []
    all_cats = []
    all_bboxes = []
    all_img_paths = []
    scene_id_map = {}

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches"):
            img_path = batch['img_path'][0]
            cat = batch['cat'][0]
            obj = batch['obj'][0]
            seg_path = batch['seg_path'][0]
            img, seg = read_data_dino(img_path, seg_path)
            feat = model(img.unsqueeze(0).cuda())
            feat = F.normalize(feat, dim=1)
            sim[idx] = F.cosine_similarity(feat, base_feat, dim=1)
            scene = img_path.split('/')[-3]
            all_objs.append(obj)
            all_cats.append(cat)
            all_img_paths.append(img_path)
            all_bboxes.append(get_bbox_from_seg(seg))
            if scene in scene_id_map.keys():
                scene_id_map[scene].append(idx)
            else:
                scene_id_map[scene] = [idx]
    ####### identify leeast sim                        
    # import pdb; pdb.set_trace()

    for scene in scene_id_map.keys():
        indices = torch.tensor(scene_id_map[scene], dtype=torch.int32)
        sim_scores = sim[indices]
        min_sim_score_, _ = torch.max(sim_scores,dim=1)
        min_sim_score_id_ = torch.argmin(min_sim_score_)
        min_sim_score_id = indices[min_sim_score_id_].cpu().numpy()
        min_obj = np.array(all_objs)[min_sim_score_id]
        print(min_obj)
        min_cat = np.array(all_cats)[min_sim_score_id]

        min_bbox = all_bboxes[min_sim_score_id]
        img_path = all_img_paths[min_sim_score_id]
        if min_bbox is None:
            min_bbox = []
        else:
            min_bbox = [int(min_bbox[0]), int(min_bbox[1]), int(min_bbox[2]), int(min_bbox[3])]
        all_preds.append({'bbox':min_bbox , 'img_path': img_path, 'cat': min_cat, 'obj': min_obj, 'response': '', 'prompt': ''})
        # for ind, index in enumerate(indices):
        #     if ind != min_sim_score_id_:
        #         pred_ind = torch.argmax(sim[index]).cpu().numpy()
        #         pred_obj = np.array(all_objs)[pred_ind]
        #         pred_cat = np.array(base_label)[pred_ind]
        #         pred_bbox = np.array(all_bboxes)[pred_ind]
        #         all_preds.append({'bbox': pred_bbox, 'img_path': img_path, 'cat': pred_cat, 'obj': pred_obj, 'response': '', 'prompt': ''})

    with open(os.path.join('/'.join(img_path.split('/')[:-3]), f'{args.model}_{args.data_mode}_{args.load_seg}_pred_{args.run}.json'), 'w') as f:
        json.dump(all_preds, f, indent=4)
    




def main():
    # seed = args.seed

    # torch.manual_seed(seed)

    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

    # np.random.seed(seed)
    # random.seed(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    dataset = BasicDataset(mode=args.data_mode, seed=args.seed, num_views=args.num_views)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        pin_memory=True
    )
    all_preds = []
    model = get_model(args.model, args)


    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches"):
            img_path = batch['img_path'][0]
            cat = batch['cat'][0]
            obj = batch['obj'][0]
            mask = False
            if args.data_mode == 'known_given' or args.data_mode == 'known_given_mask':
                if 'mask' in args.data_mode:
                    mask = True
                other_bboxes, other_cats = batch['other_bboxes'], batch['other_cats']
            else:
                other_bboxes, other_cats = None, None
            if args.data_mode == '2_unknowns':
                scene_graph = batch['scene_graph'][0]
            else:
                scene_graph = None
            
            prompt = model.get_prompt(cat, other_bboxes, other_cats, scene_graph)
            print(prompt)
            image = model.process_image(img_path, other_bboxes, other_cats, mask)
            # import pdb; pdb.set_trace()
            input_by_model = model.build_input_ids(image, prompt, [])
            inputs = model.get_model_input(input_by_model)
            outputs = model.generate_content(inputs)
            # if args.model == 'gemini':
            #     time.sleep(10)
            # import pdb; pdb.set_trace()

            response, bbox = model.process_model_output(inputs, outputs)
            # print(response)
            image_vis = Image.open(img_path).convert('RGB')

            # import pdb; pdb.set_trace()
            if bbox is not None:
                # import pdb; pdb.set_trace()
                if len(bbox) > 0 and type(bbox[0]) == np.ndarray:
                    ##### a mask
                    if bbox[0].shape[0] > 0:
                        try:
                            if bbox[0].shape[0] > 1:
                                bbox = [bbox[0][0]]
                            bbox = get_bbox_from_seg(bbox[0].squeeze())
                        except:
                            import pdb; pdb.set_trace()
                    else:
                        bbox = []
            if bbox is not None and len(bbox) > 0:
                # if args.model == 'cogvlm':
                #     xmin, ymin, xmax, ymax = bbox
                #     bbox = [xmin, xmax, ymin, ymax]

                # image_vis = draw_bounding_boxes(image_vis, [bbox], resized=(args.image_size, args.image_size))
                # image_vis.save(f'output_{idx}.png')
                
                bbox = get_rescaled_bbox(bbox, resized=(args.image_size, args.image_size))

            else:
                bbox = []
            all_preds.append({'bbox': bbox, 'img_path': img_path, 'cat': cat, 'obj': obj, 'response': response, 'prompt': prompt})

    with open(os.path.join('/'.join(img_path.split('/')[:-3]), f'{args.model}_{args.data_mode}_pred_{args.run}.json'), 'w') as f:
        json.dump(all_preds, f, indent=4)

args = parse_args()

if args.model.startswith('dino'):
    main_dino()
else:
    main()