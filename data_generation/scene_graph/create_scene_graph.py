import numpy as np
import json
from PIL import Image
import os
from pathlib import Path
from joblib import Parallel, delayed 
import argparse

parser = argparse.ArgumentParser()                                                         
parser.add_argument('--data_path', type=str, help='path to dataset to overlay backgrounds')
parser.add_argument('--out_path', type=str, help='path to dataset to overlay backgrounds')


args = parser.parse_args()  

src_dataset_path = args.data_path 

rgb_file_paths = []
subdirs = [sd for sd in os.listdir(src_dataset_path) if not sd.endswith('npz')]


file_paths = [os.path.join(src_dataset_path, d) for d in subdirs if not d.endswith('npy') and not d.endswith('npz') and not d.endswith('.json')  if d != 'split' ]

print(len(file_paths))


def get_intrinsics():
    K = np.array([[186.6666717529297, 0.0, 112.0], [0.0, 186.6666717529297, 112.0], [0.0, 0.0, 1.0]])
    return K

def extract_intrinsics(K):
    """
    Extract fx, fy, cx, cy from the camera intrinsic matrix K.

    Args:
        K (np.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        fx, fy, cx, cy (float): Extracted intrinsics.
    """
    fx = K[0, 0]  # Focal length in x
    fy = K[1, 1]  # Focal length in y
    cx = K[0, 2]  # Principal point x
    cy = K[1, 2]  # Principal point y
    return fx, fy, cx, cy

def project_to_3d(u, v, depth, intrinsics):
    """Convert 2D pixel (u,v) and depth to 3D world coordinates."""
    fx, fy, cx, cy = intrinsics
    Z = depth[v, u]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])

def read_segmentation(filename):
    img = Image.open(filename).convert("L") 
    img = np.array(img)
    img = img > 128
    return img.astype(np.uint8)

def read_depth(filename):
    img = np.load(filename, allow_pickle=True)['img']
    return img

def get_bbox_from_seg(segmentation):
    """Given a binary segmentation mask, compute the bounding box [xmin, ymin, xmax, ymax]."""
    rows = np.any(segmentation, axis=1)
    cols = np.any(segmentation, axis=0)

    if not np.any(rows) or not np.any(cols):  # No object found
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return [xmin, ymin, xmax, ymax]

def compute_iou(box_A, box_B):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_A (dict): {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}
        box_B (dict): {'x_min': xmin, 'x_max': xmax, 'y_min': ymin, 'y_max': ymax}

    Returns:
        float: IoU value in range [0, 1].
    """

    # Compute intersection coordinates
    x_min_inter = max(box_A['x_min'], box_B['x_min'])
    y_min_inter = max(box_A['y_min'], box_B['y_min'])
    x_max_inter = min(box_A['x_max'], box_B['x_max'])
    y_max_inter = min(box_A['y_max'], box_B['y_max'])

    # Compute width and height of intersection
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)

    # Compute intersection area
    intersection_area = inter_width * inter_height

    # Compute areas of both bounding boxes
    area_A = (box_A['x_max'] - box_A['x_min']) * (box_A['y_max'] - box_A['y_min'])
    area_B = (box_B['x_max'] - box_B['x_min']) * (box_B['y_max'] - box_B['y_min'])

    # Compute union area
    union_area = area_A + area_B - intersection_area

    # Compute IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou

def compute_relationships(objects_3d, bboxes, thres=0.0):
    """Compute spatial relationships using both centroids and bounding boxes."""
    relationships = {obj: {} for obj in objects_3d}

    for obj_a, pos_a in objects_3d.items():
        bbox_a = bboxes[obj_a]
        for obj_b, pos_b in objects_3d.items():
            if obj_a == obj_b:
                continue
            # import pdb; pdb.set_trace()

            bbox_b = bboxes[obj_b]

            # Centroid-based relationships
            # if np.linalg.norm(pos_a - pos_b) < 1.0:
            #     relationships[obj_a].setdefault("next to", []).append(obj_b)

            # Bounding box-based relationships
            if bbox_a["x_max"] <= bbox_b["x_min"]:
                relationships[obj_a].setdefault("to the left of", []).append(obj_b)
            elif bbox_a["x_min"] >= bbox_b["x_max"]:
                relationships[obj_a].setdefault("to the right of", []).append(obj_b)

            iou = compute_iou(bbox_a, bbox_b)
            # print(obj_a, obj_b, iou)

            diff = pos_b - pos_a
            x_diff, y_diff, z_diff = diff


            if z_diff > 0 and iou > thres:
                relationships[obj_a].setdefault("in front of", []).append(obj_b)
            elif z_diff < 0 and iou > thres:
                relationships[obj_a].setdefault("behind", []).append(obj_b)

    return relationships

def get_scene_graph(path):
    """Given a path to a directory containing segmentation masks and depth maps, compute a scene graph."""
    intrinsics = get_intrinsics()
    intrinsics = extract_intrinsics(intrinsics)
    objects_3d = {}
    bboxes = {}
    seg_dir = Path(path) / 'segmentations'
    scene_relationships = {}
    for view in range(25):
        for obj in os.listdir(seg_dir):

            seg_path = seg_dir / obj / f'{obj}_{view:04d}.png'
            depth_path = Path(path) / 'depth_NPZ' / f'{view:04d}.npz'

            segmentation = read_segmentation(seg_path)
            depth = read_depth(depth_path)

            if np.sum(segmentation) < 30:
                continue
            bbox = get_bbox_from_seg(segmentation)
            if bbox is None:
                continue

            v, u = np.mean(np.where(segmentation), axis=1).astype(np.int32)
            pos = project_to_3d(u, v, depth, intrinsics)

            objects_3d[obj] = pos
            bboxes[obj] = {
                "x_min": bbox[0],
                "y_min": bbox[1],
                "x_max": bbox[2],
                "y_max": bbox[3]
            }
        # print('--------')
        # if view == 3:
        #     import pdb; pdb.set_trace()
        
        view_relationships = compute_relationships(objects_3d, bboxes)
        scene_relationships[view] = view_relationships

    return scene_relationships

def job(arg):
    spath = arg
    scene_relationships = get_scene_graph(spath)

    return scene_relationships

results = Parallel(n_jobs=12, verbose=1, backend="multiprocessing")(map(delayed(job), file_paths))
final_data = {}
for (sg, path) in zip(results, file_paths):
    final_data[path] = sg

with open(args.out_path, 'w') as f:
    json.dump(final_data, f, indent=4)

import pdb; pdb.set_trace()
print('haah')
# scene_relation_ship = get_scene_graph(file_paths[100])
# print(scene_relation_ship)
