import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib

from prettytable import PrettyTable

matplotlib.use("Agg")

def show_mask(mask, ax=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.reshape(h,w,1)
    mask_image = mask * color.reshape(1, 1, -1)
    return mask_image[:,:,:3]
    # ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

def get_box(seg):
    ########## Seg returned from read_segmentation
    img_size = seg.shape

    seg1 = np.sum(seg, axis=0)
    seg2 = np.sum(seg, axis=1)

    cmin = np.where(seg1 > 0)[0][0]
    rmin = np.where(seg2 > 0)[0][0]

    cmax = np.where(seg1 > 0)[0][-1]
    rmax = np.where(seg2 > 0)[0][-1]

    return rmin, rmax, cmin, cmax


def pad_segs(gt_seg, num_queries):
    B, N, H, W = gt_seg.shape
    if N < num_queries:
        padded_segs = torch.cat([gt_seg, torch.zeros(B,num_queries-N,H,W).cuda()], dim=1)
    else:
        padded_segs = gt_seg
    return padded_segs

import re
def extract_bounding_box(output_string, factor=1.0):
    """
    Extracts bounding box coordinates from a given string.
    
    Args:
        output_string (str): The input string containing the bounding box in the format [xmin, xmax, ymin, ymax].

    Returns:
        list: A list of four integers representing [xmin, xmax, ymin, ymax], or None if not found.
    """
    # Use regex to find the bounding box pattern in the format [xmin, xmax, ymin, ymax]
    if factor != 1.0:
        match = re.search(r"\[\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*\]", output_string)
    else:
        match = re.search(r"\[\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*\]", output_string)
    if match:
        # Convert extracted values to integers and return as a list
        if factor != 1.0:
            arr = map(float, [match.group(1), match.group(3), match.group(5), match.group(7)])
            print(arr)
            arr = [int(x*factor) for x in arr]
        else:
            arr = [int(match.group(i)) for i in range(1, 5)]

        return arr
    else:
        return None  # Return None if no bounding box is found
