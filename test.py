import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import argparse

NOVEL_OBJS_PATH = '/home/ant/develop/LSME/data_generation/common/jsons/geoshape20_dict.json'


with open(NOVEL_OBJS_PATH, 'r') as f:
    novel_objs = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help='model name')
parser.add_argument("--data_mode", type=str, default="most_visible_object", help='data mode')
parser.add_argument("--classes", type=str, default="novel", help='data mode')

parser.add_argument("--data_path", type=str, default="", help='path to directory')
parser.add_argument("--criteria", type=str, default="", help='path to directory')

parser.add_argument("--save_pred", type=bool, default=False, help='path to directory')
parser.add_argument("--n_run", type=int, default=1, help='path to directory')
parser.add_argument("--num_known", type=int, default=1, help='path to directory')
parser.add_argument("--title", type=str, default="", help='path to directory')
parser.add_argument("--variant", type=str, default="", help='path to directory')
parser.add_argument("--load_seg", type=str, default="", help='path to directory')



plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14 # Set global font size


args = parser.parse_args()
model = args.model
data_mode = args.data_mode
data_path = args.data_path

novel_object_names = [
    "dax", "wug", "blicket", "zorp", "flib", "toma", "kiki", "gorp",
    "snod", "lirp", "plon", "zib", "trop", "skib", "mib", "glorp",
    "fepe", "norg", "quim", "vimp", "sprock", "chiz", "tropin", "blick",
    "zibble", "gloop", "quorp", "thag", "vark", "snib", "florp", "crat"
]

evaluated_cats = novel_object_names

import matplotlib.pyplot as plt
import numpy as np

def collect_criteria(data_path, model, data_mode, n_run, criteria, thres=[0.5], num_known=1):
    if criteria == 'match_novel':
        percentage = analyze_correct_novel_given_correct_known_by_thres(data_path, model, \
                                    data_mode, int(n_run), thres, num_known)
    else:
        percentage = analyze_novel_match_known_given_correct_known_by_thres(data_path, model, \
                                        data_mode, int(n_run), criteria, thres, num_known)
    return np.mean(percentage, axis=0) #### getting average of runs

def plot_bar_chart(methods, categories, data):
# Define colors for each component
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'm', 'brown']

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot each category stacked on top of each other
    bottom = np.zeros(len(methods))  # Initialize bottom position for stacking
    for i, (category, color) in enumerate(zip(categories, colors)):
        ax.bar(methods, data[:, i], bottom=bottom, label=category, color=color, alpha=0.5)
        bottom += data[:, i]  # Update bottom for stacking

    # Labels and title
    ax.set_ylabel("Probability")
    ax.set_xlabel("Methods")
    # ax.set_title("ME Analysis on 2K-1U-YesD with Obj. Names & Masks")
    ax.set_title("ME Analysis on 1 Known-1 Unknown Object Setting")

    ax.set_ylim(0, 1)  # Ensure y-axis is from 0 to 1
    ax.set_xticklabels(methods, fontsize=12)
    ax.legend(title="Model Predictions", loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(categories), fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Display percentages on each section of the bars
    for i in range(len(methods)):
        cumulative = 0
        for j in range(len(categories)):
            value = data[i, j]
            ax.text(i, cumulative + value / 2, f"{value * 100:.1f}%", ha='center', va='center', fontsize=10, color="black")
            cumulative += value
    plt.tight_layout()
    # Show plot
    # plt.savefig('2K-1U-barchart-kg_masks_2.pdf')
    plt.savefig('1K-1U-barchart-slides.pdf')


def read_segmentation(path):
    seg = Image.open(path).convert('L')
    seg = np.array(seg) > 128
    return seg.astype(np.uint8)

def get_bbox_from_seg(seg):
    # Find non-zero (foreground) pixel coordinates
    y_indices, x_indices = np.where(seg > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None  # No object found in the mask

    # Compute bounding box
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    ymin, ymax = np.min(y_indices), np.max(y_indices)

    return [xmin, xmax, ymin, ymax]

def compute_iou(bbox_gt, bbox_pred, model):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox_gt (list or np.ndarray): Ground-truth bounding box [xmin, ymin, xmax, ymax].
        bbox_pred (list or np.ndarray): Predicted bounding box [xmin, ymin, xmax, ymax].

    Returns:
        float: IoU value between 0 and 1. Returns 0 if there is no intersection.
    """
    # Extract coordinates
    x1_gt, x2_gt, y1_gt, y2_gt = bbox_gt
    if model != 'cogvlm':
        x1_pred, x2_pred, y1_pred, y2_pred = bbox_pred
    else:
        x1_pred, y1_pred, x2_pred, y2_pred = bbox_pred

    # Compute intersection rectangle
    x1_inter = max(x1_gt, x1_pred)
    y1_inter = max(y1_gt, y1_pred)
    x2_inter = min(x2_gt, x2_pred)
    y2_inter = min(y2_gt, y2_pred)

    # Compute area of intersection
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Compute area of both bounding boxes
    gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)

    # Compute union area
    union_area = gt_area + pred_area - inter_area

    # Compute IoU (handle case where union_area is 0)
    iou = inter_area / union_area if union_area > 0 else 0.0

    return iou

def draw_bounding_boxes(image, bounding_boxes_with_labels, model, resized=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = np.array(image)

    for id, bounding_box in enumerate(bounding_boxes_with_labels):

        # Normalize the bounding box coordinates
        width, height = image.shape[1], image.shape[0]
        if model != 'cogvlm' or id == 1:
            xmin, xmax, ymin, ymax = bounding_box
        elif model == 'cogvlm' and id == 0:
            xmin, ymin, xmax, ymax = bounding_box
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
        
        if id == 1:
            color = [0, 255, 0]
        else:
            color = [255, 0, 0]

        # color = np.random.randint(0, 256, (3,)).tolist()

        box_thickness = 2


        # cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, -1)
        # cv2.putText(image, label, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)

    image = Image.fromarray(image)
    return image

def plot_map_curve(all_average_precisions, all_labels, thresholds, \
                   savefigname, title="CogVLM mAP Curve on Novel Objects", \
                    num_run=1, xlabel='IoU Thresholds', ylabel='Average Precision',\
                    ylim=[0,1]):
    """
    Plots the mAP (Mean Average Precision) curve and annotates exact values at each point.

    Args:
        average_precisions (list or np.ndarray): List of average precision values.
        thresholds (list or np.ndarray): Corresponding threshold values.
        title (str): Title of the plot (default: "mAP Curve").
    """
    # Convert to NumPy arrays for safety
    colors = ['b', 'r', 'black', 'g', 'gold', 'm', 'c', 'olive', 'brown', 'orange', 'lime']
    np.random.seed(2025)
    # colors = np.random.rand(len(all_labels),3)
    plt.figure(figsize=(8, 5))
    print(thresholds)

    for i, average_precisions in enumerate(all_average_precisions):
        mean_precisions = np.mean(average_precisions, axis=0)
        std_precisions = np.std(average_precisions, axis=0)
        thresholds = np.array(thresholds)

        # Ensure sorting by threshold values
        sorted_indices = np.argsort(thresholds)
        thresholds = thresholds[sorted_indices]
        mean_precisions = mean_precisions[sorted_indices]
        print(mean_precisions)

        # Plot mAP curve
        plt.plot(thresholds, mean_precisions, marker="o", linestyle="-", color=colors[i], label=all_labels[i])
        plt.errorbar(thresholds, mean_precisions, yerr=std_precisions, fmt='o', color=colors[i])
        print_ap(all_labels[i], mean_precisions, 0.5, thresholds)
        print_ap(all_labels[i], mean_precisions, 0.75, thresholds)
        print('-------------')


        # Annotate each point with its exact value
        # for x, y in zip(thresholds, average_precisions):
        #     plt.text(x, y, f"{y:.2f}", fontsize=9, ha="left", va="bottom", color="black")

    # Formatting the plot
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=18)
    plt.grid(True)
    if len(all_labels) < 8:
        legend = plt.legend(ncol=2, fontsize=14)
    else:
        legend = plt.legend(ncol=3, fontsize=12)

    plt.ylim(ylim[0], ylim[1])
    
    frame = legend.get_frame()

    # Set the alpha value of the frame
    frame.set_alpha(0.5)  # Set alpha to 0.5 (50% transparency)
    plt.savefig(savefigname)

def read_json(data_path, model, data_mode, run, load_seg='gt'):
    # import pdb; pdb.set_trace()
    if args.variant == '2K-1U-YesD':
        pred = 'pred2'
    else:
        pred = 'pred'
    if not model.startswith('dino'):
        json_path = Path(data_path) / f"{model}_{data_mode}_{pred}_{run}.json"
    else:
        json_path = Path(data_path) / f"{model}_{data_mode}_{load_seg}_{pred}_{run}.json"

    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_all_ious_from_pred(data, classes, model, data_mode):
    ious = []
    match_objects = []
    empty_boxes = []
    # import pdb; pdb.set_trace()

    for d in data:
        try:
            img_path = Path(d['img_path'])
            cat = d['cat']
            if classes == 'known':
                if cat in evaluated_cats:
                    continue
            if classes == 'novel':
                if cat not in evaluated_cats:
                    continue
            obj_orig = d['obj']
            bbox_pred = d['bbox']
            base_dir = img_path.parent.parent
            filename = img_path.name
            # import pdb; pdb.set_trace()
            ######################### for cases where different objs from same cat are present
            seg_dir = base_dir / 'segmentations'
            seg_objs = [o for o in os.listdir(seg_dir)]
            max_obj_iou = 0
            obj_max = None
            if len(bbox_pred) == 0:
                empty_boxes.append(True)
            else:
                empty_boxes.append(False)
            for obj in seg_objs:
                seg_path = base_dir / 'segmentations' /obj / f"{obj}_{filename}"
                
                seg = read_segmentation(seg_path)
                bbox_gt = get_bbox_from_seg(seg)
                img = Image.open(img_path).convert('RGB')
                if len(bbox_pred) == 0:
                    iou = 0
                else:
                    if bbox_gt == None:
                        continue
                    iou = compute_iou(bbox_gt, bbox_pred, model)
                if iou > max_obj_iou:
                    max_obj_iou = iou
                    obj_max = obj

                # import pdb; pdb.set_trace()
                if args.save_pred:
                    save_folder = f"{model}_{data_mode}_{classes}_2"
                    os.makedirs(save_folder, exist_ok=True)
                    img = draw_bounding_boxes(img, [bbox_pred, bbox_gt], model)
                    img.save(os.path.join(save_folder, f'{cat}_{obj}_{filename}'))
            iou = max_obj_iou
            ious.append(iou)
            match_objects.append(obj_max)
        except:
            import pdb; pdb.set_trace()
    ious = np.array(ious)
    match_objects = np.array(match_objects)
    return ious, match_objects, np.array(empty_boxes)

def get_ious_from_pred_dino(data, classes, model, data_mode, run=0):
    # import pdb; pdb.set_trace()
    ious = []
    for d in data:
        img_path = Path(d['img_path'])
        cat = d['cat']
        obj = d['obj']
        bbox_pred = d['bbox']
        base_dir = img_path.parent.parent
        filename = img_path.name
        seg_dir = base_dir / 'segmentations'
        novel_seg_obj = [o for o in os.listdir(seg_dir) if o.startswith(tuple(novel_objs))][0]
        seg_path = base_dir / 'segmentations' / novel_seg_obj / f"{novel_seg_obj}_{filename}"
        seg = read_segmentation(seg_path)
        bbox_gt = get_bbox_from_seg(seg)
        if len(bbox_pred) == 0:
            iou = 0
        else:
            if bbox_gt == None:
                continue
            iou = compute_iou(bbox_gt, bbox_pred, model)
        ious.append(iou)
    ious = np.array(ious)
    # import pdb; pdb.set_trace()

    return ious

def get_ious_from_pred(data, classes, model, data_mode, run=0):
    if model.startswith('dino'):
        return get_ious_from_pred_dino(data, classes, model, data_mode, run=0)
    ious = []
    # import pdb; pdb.set_trace()

    for d in data:
        # try:
        img_path = Path(d['img_path'])
        cat = d['cat']
        if classes == 'known':
            if cat in evaluated_cats:
                continue
        if classes == 'novel':
            if cat not in evaluated_cats:
                continue
        obj = d['obj']
        bbox_pred = d['bbox']
        base_dir = img_path.parent.parent
        filename = img_path.name
        # import pdb; pdb.set_trace()
        ######################### for cases where different objs from same cat are present
        seg_dir = base_dir / 'segmentations'
        seg_objs = [o for o in os.listdir(seg_dir) if (o.startswith(cat) \
                                    if classes == 'known' else o.startswith(obj))]
        max_obj_iou = 0
        for obj in seg_objs:
            seg_path = base_dir / 'segmentations' /obj / f"{obj}_{filename}"
            
            seg = read_segmentation(seg_path)
            bbox_gt = get_bbox_from_seg(seg)
            img = Image.open(img_path).convert('RGB')
            if len(bbox_pred) == 0:
                iou = 0
            else:
                if bbox_gt == None:
                    continue
                iou = compute_iou(bbox_gt, bbox_pred, model)
            max_obj_iou = max(max_obj_iou, iou)

            # import pdb; pdb.set_trace()
            if args.save_pred:
                save_folder = f"{model}_{data_mode}_{classes}_r{str(run)}"
                os.makedirs(save_folder, exist_ok=True)
                img = draw_bounding_boxes(img, [bbox_pred, bbox_gt], model)
                img.save(os.path.join(save_folder, f'{cat}_{obj}_{filename}'))
        iou = max_obj_iou
        ious.append(iou)
        # except Exception as e:
        #     import pdb; pdb.set_trace()
        #     print(e)
    ious = np.array(ious)
    return ious

def get_data_index(data, classes):
    return np.array([i for i in range(len(data)) if (data[i]['cat'] not in evaluated_cats \
                                            if classes == 'known' else data[i]['cat'] in evaluated_cats)])

def analyze_correct_novel_given_correct_known_by_thres(data_path, model, data_mode, n_run, thresholds=0.5, num_known=1):
    # import pdb; pdb.set_trace()
    all_percentages = []
    for run in range(n_run):
        data = read_json(data_path, model, data_mode, run)
        index_novel = get_data_index(data, 'novel')

        index_known = get_data_index(data, 'known')
        iou_novel = get_ious_from_pred(data, 'novel', model, '')

        iou_known = get_ious_from_pred(data, 'known', model, '')
        data_known = [data[i] for i in index_known]
        data_novel = [data[i] for i in index_novel]
        all_percentages_thres = []
        for thres in thresholds:
            correct_known_index = np.where(iou_known > thres)[0]
            data_known_correct = [data_known[i] for i in correct_known_index]
            img_path_known_correct = np.array([d['img_path'] for d in data_known_correct])
            # import pdb; pdb.set_trace()

            if num_known > 1:
                #### more than 1 known obj
                unique_arr, unique_count = np.unique(img_path_known_correct, return_counts=True)
                img_path_known_correct = unique_arr[unique_count == num_known]
            
            index_novel_known_correct = np.array([i for i in range(len(data_novel)) \
                                        if data_novel[i]['img_path'] in img_path_known_correct])

            if len(index_novel_known_correct) > 0:
                iou_novel_known_correct = iou_novel[index_novel_known_correct]
                percentage_novel_correct = len(iou_novel_known_correct[iou_novel_known_correct > thres]) / len(index_novel_known_correct)

            else:
                percentage_novel_correct = 0
            all_percentages_thres.append(percentage_novel_correct)
        all_percentages.append(all_percentages_thres)
    return all_percentages

def analyze_novel_match_known_given_correct_known_by_thres(data_path, model, data_mode, n_run, criteria='match_known', thresholds=0.5, num_known=1):
    # import pdb; pdb.set_trace()
    all_percentages = []
    for run in range(n_run):
        data = read_json(data_path, model, data_mode, run)
        index_novel = get_data_index(data, 'novel')

        index_known = get_data_index(data, 'known')
        iou_novel, match_obj_novel, empty_boxes = get_all_ious_from_pred(data, 'novel', model, '')
        iou_correct_novel = get_ious_from_pred(data, 'novel', model, '')

        iou_known = get_ious_from_pred(data, 'known', model, '')
        data_known = [data[i] for i in index_known]
        data_novel = [data[i] for i in index_novel]
        all_percentages_thres = []
        for thres in thresholds:
            correct_known_index = np.where(iou_known > thres)[0]
            data_known_correct = [data_known[i] for i in correct_known_index]
            img_path_known_correct = np.array([d['img_path'] for d in data_known_correct])
            if num_known > 1:
                #### more than 1 known obj
                unique_arr, unique_count = np.unique(img_path_known_correct, return_counts=True)
                img_path_known_correct = unique_arr[unique_count == num_known]
            index_novel_known_correct = np.array([i for i in range(len(data_novel)) \
                                        if data_novel[i]['img_path'] in img_path_known_correct])

            if len(index_novel_known_correct) > 0:
                iou_novel_known_correct = iou_novel[index_novel_known_correct]
                match_obj_novel_temp = match_obj_novel[index_novel_known_correct]
                empty_boxes_temp = empty_boxes[index_novel_known_correct]
                index_novel_match = np.where(iou_novel_known_correct > thres)[0]
                match_obj_novel_temp = match_obj_novel_temp[index_novel_match]

                iou_correct_novel_known_correct = iou_correct_novel[index_novel_known_correct]
                index_correct_novel_match = np.where(iou_correct_novel_known_correct > thres)[0]
                ############# novel match with known
                if criteria == 'match_known':
                    match_obj_novel_temp = [mo for mo in match_obj_novel_temp if not mo.startswith(tuple(novel_objs))]
                    percentage_novel_match_unknown = len(match_obj_novel_temp) / len(index_novel_known_correct)
                elif criteria == 'empty':
                    percentage_novel_match_unknown = np.sum(empty_boxes_temp) / len(empty_boxes_temp)
                elif criteria == 'match_bg':
                    # import pdb; pdb.set_trace()
                    percentage_novel_match_unknown = (len(index_novel_known_correct) - len(index_novel_match) - np.sum(empty_boxes_temp)) / len(index_novel_known_correct)
                elif criteria == 'match_other_novel':
                    match_obj_novel_temp = [mo for mo in match_obj_novel_temp if mo.startswith(tuple(novel_objs))]

                    percentage_novel_match_unknown = (len(match_obj_novel_temp) - len(index_correct_novel_match)) / len(index_novel_known_correct)
            else:
                percentage_novel_match_unknown = 0
            all_percentages_thres.append(percentage_novel_match_unknown)
        all_percentages.append(all_percentages_thres)
    return all_percentages


def collect_aps_by_thresholds(ious, x_axis):

    all_results = []
    for thres in x_axis:
        results = np.mean(ious >= thres)
        all_results.append(results)
    return all_results

def get_each_model_aps(data_path, model, data_mode, classes, thres, run, load_seg):
    # import pdb; pdb.set_trace()
    data = read_json(data_path, model, data_mode, run, load_seg)
    ious = get_ious_from_pred(data, classes, model, data_mode, run)

    aps = collect_aps_by_thresholds(ious, thres)
    return aps

def plot_models_data_modes(all_models, all_labels, data_modes, classes, data_path, thres, savefile, title, n_run, load_seg):

    all_model_results = []
    count = 0
    to_remove = []

    for i,data_mode in enumerate(data_modes):
        for j, model in enumerate(all_models):
            try:
                all_runs = []

                for r in range(n_run):
                    # import pdb; pdb.set_trace()
                    
                    aps = get_each_model_aps(data_path, model, data_mode, classes,thres, str(r), load_seg[j])
                    all_runs.append(aps)
                all_model_results.append(all_runs)
                    
            except:

                to_remove.append(count)
            count += 1
    # print(to_remove)
    for tr in to_remove:
        all_labels.pop(tr)
    plot_map_curve(all_model_results, all_labels, thres, savefile, title=title, num_run=n_run)

def plot_models_me(all_models, all_labels, data_modes, data_path, thres, savefile, title, n_run, criteria, num_known):
    all_model_results = []
    for data_mode in data_modes:
        for model in all_models:

            if criteria == 'match_novel':
                percentage = analyze_correct_novel_given_correct_known_by_thres(data_path, model, \
                                            data_mode, n_run, thres, num_known)
            else:
                percentage = analyze_novel_match_known_given_correct_known_by_thres(data_path, model, \
                                        data_mode, n_run, criteria, thres, num_known)
            all_model_results.append(percentage)
    print(all_model_results)

    plot_map_curve(all_model_results, all_labels, thres, \
                   savefile, title=title, num_run=n_run, ylabel='Percentage')
    
def get_ap_at_thres(data, thres, all_thres):
    return data[all_thres ==  thres]

def print_ap(model, data, thres, all_thresholds):
    print(f"{model} - AP@{str(thres)}: {str(get_ap_at_thres(data, thres, all_thresholds))}")
        

def main():
    if args.model == '':
        if len(args.criteria) == 0:
            if (args.variant == '2K-1U-YesD' or args.variant == '1K-1U-YesD' )and args.classes == 'novel':
                all_models = ['cogvlm', 'sa2va', 'omg_llava', 'llava_ov', 'gemini', 'lisa', 'flmm', 'dino', 'dinov1', 'dino', 'dinov1']
                all_load_seg = ['', '', '', '', '', '', '', 'gt', 'gt', 'sam2_masks', 'sam2_masks']
            else:
                # all_models = ['cogvlm', 'sa2va', 'omg_llava', 'llava_ov', 'gemini', 'lisa', 'flmm']
                # all_load_seg = ['', '', '', '', '', '', '', 'gt', 'gt', 'sam2_masks', 'sam2_masks']
                all_models = ['cogvlm', 'sa2va', 'omg_llava', 'gemini', 'lisa', 'flmm']
                all_load_seg = ['', '', '', '', '', '', '', 'gt', 'gt', 'sam2_masks', 'sam2_masks']



        else:
            # all_models = ['cogvlm', 'sa2va', 'omg_llava', 'gemini', 'lisa', 'flmm']
            all_load_seg = ['', '', '', '', '', '', '', 'gt', 'gt', 'sam2_masks', 'sam2_masks']

            all_models = ['cogvlm', 'flmm', 'lisa', 'sa2va', 'gemini', 'omg_llava']



        if len(args.title) == 0:
            title = 'all models'
        savefile = 'all_models'
    else:
        all_models = [args.model]
        title = f"{args.model}"
        savefile = f"{args.model}"
        all_load_seg = ['', '', '', '', '', '', '', 'gt', 'gt', 'sam2_masks', 'sam2_masks']

    if args.data_mode == '':
        # data_modes = ['most_visible_object', 'visible_object', 'known_given', 'most_visible_novel_object']
        data_modes = ['most_visible_object', 'visible_object']
        # data_modes = ['2_unknowns', '2_unknowns_no_sg']


        title = f"{title} all data modes {args.classes}"
        savefile = f"{savefile}_all_data_modes_{args.classes}.png"
    else:
        data_modes = [args.data_mode]
        if len(args.title) == 0:
            title = f"{title} {args.data_mode}_{args.classes}"
        savefile = f"{savefile}_{args.data_mode}_{args.classes}_{args.variant}.png"
    if len(all_models) == 1:
        all_labels = data_modes
    if len(data_modes) ==1 :
        all_labels = all_models
    if len(all_models) > 1 and len(data_modes) > 1:
        all_labels = [f"{model}_{data_mode}" for data_mode in data_modes for model in all_models]
        print(all_labels)

    if args.title != '':
        title = args.title
    if len(args.criteria) == 0:
        # all_labels = ['CogVLM', 'Sa2VA', 'OMG-LLaVA', 'LLaVA-OV', 'Gemini', 'LISA', 'F-LMM', 'DINOv2-GT', 'DINOv1-GT', 'DINOv2-SAM2', 'DINOv1-SAM2']
        # all_labels = ['CogVLM', 'Sa2VA', 'OMG-LLaVA', 'Gemini', 'LISA', 'F-LMM']
        all_labels = ['CogVLM', 'F-LMM', 'LISA', 'Sa2VA', 'Gemini', 'OMG-LLaVA']

    if len(args.criteria) == 0:
        plot_models_data_modes(all_models, all_labels, data_modes, args.classes, data_path, np.linspace(0.1, 0.95, 18), savefile, title, args.n_run, all_load_seg)
    elif args.criteria == 'combined':
        # import pdb; pdb.set_trace()
        criterias = ['match_novel', 'match_known', 'empty', 'match_bg']
        if args.variant == '1K-2U-YesD':
            criterias.append('match_other_novel')

        data = np.zeros((len(all_models), len(criterias)))
        for mi, model in enumerate(all_models):

            for ci, criteria in enumerate(criterias):
                data[mi,ci] = collect_criteria(data_path, model, data_mode, args.n_run, criteria, [0.5], args.num_known)
        if args.variant == '1K-2U-YesD':
            criteria_labels = ['N->N', 'N->K', 'No Prediction', 'N->Bg', 'N->NO']
        else:
            criteria_labels = ['N->N', 'N->K', 'No Prediction', 'N->Bg']
        all_labels = ['CogVLM', 'Sa2VA', 'OMG-LLaVA','Gemini', 'LISA', 'F-LMM']
        plot_bar_chart(all_labels, criteria_labels, data)
    
    else:
        plot_models_me(all_models, all_labels, data_modes, data_path, np.linspace(0.5, 0.9, 9), savefile, title, args.n_run, args.criteria, args.num_known)
    


# def main():
#     all_model_results = []
#     if args.model == '':
#         all_models = ['cogvlm', 'sa2va', 'omg_llava', 'glamm', 'llava_ov']
#         savefile = f"all_{data_mode}_{args.classes}.png"
#         title = f"all {data_mode} {args.classes} classes"
#     else:
#         all_models = [args.model]
#         savefile = f"{args.model}_{data_mode}_{args.classes}.png"
#         title = f"{args.model} {data_mode} {args.classes} classes"
#     for model in all_models:
#         ious = []

#         json_path = Path(data_path) / f"{model}_{data_mode}_pred.json"
#         with open(json_path, 'r') as f:
#             data = json.load(f)
#         for d in data:
#             img_path = Path(d['img_path'])
#             cat = d['cat']
#             if args.classes == 'known':
#                 if cat in evaluated_cats:
#                     continue
#             if args.classes == 'novel':
#                 if cat not in evaluated_cats:
#                     continue
#             obj = d['obj']
#             bbox_pred = d['bbox']
#             base_dir = img_path.parent.parent
#             filename = img_path.name
#             seg_path = base_dir / 'segmentations' /obj / f"{obj}_{filename}"
            
#             seg = read_segmentation(seg_path)
#             bbox_gt = get_bbox_from_seg(seg)
#             img = Image.open(img_path).convert('RGB')
#             # import pdb; pdb.set_trace()
#             if len(bbox_pred) == 0:
#                 iou = 0
#             else:
#                 # img = draw_bounding_boxes(img, [bbox_pred, bbox_gt])
#                 # img.save(f'{cat}_{obj}_{filename}')
#                 iou = compute_iou(bbox_gt, bbox_pred, model)
#             ious.append(iou)
#         ious = np.array(ious)
#         # print(ious)
#         ##### plot
#         x_axis = np.linspace(0.1, 0.9, 9)
#         all_results = []
#         for thres in x_axis:
#             results = np.mean(ious >= thres)
#             all_results.append(results)
#         print(all_results)
#         all_model_results.append(all_results)
#     plot_map_curve(all_model_results, all_models, x_axis, savefile, title=title)

main()
     