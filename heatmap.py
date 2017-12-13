from scipy.ndimage.measurements import label
import numpy as np
from bounding_box import BoundingBox

def cluster_bounding_boxes(image,bounding_boxes,threshold):
    # draw_img = np.copy(image)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat,bounding_boxes)
    heat = apply_threshold(heat,threshold)
    heatmap = np.clip(heat,0,255)
    labels = get_labels(heatmap)
    bboxes = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        x_c = (np.max(nonzerox) + np.min(nonzerox)) / 2
        y_c = (np.max(nonzeroy) + np.min(nonzeroy)) / 2
        w = np.max(nonzerox) - np.min(nonzerox)
        h = np.max(nonzeroy) - np.min(nonzeroy)
        bboxes.append(BoundingBox([x_c,y_c,w,h]))
    return bboxes

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def get_labels(heatmap):
    return label(heatmap)