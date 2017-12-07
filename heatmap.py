from scipy.ndimage.measurements import label
import numpy as np
import visualizer

def cluster_bounding_boxes(image,bounding_boxes,threshold):
    draw_img = np.copy(image)
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat,bounding_boxes)
    heat = apply_threshold(heat,threshold)
    heatmap = np.clip(heat,0,255)
    labels = get_labels(heatmap)
    draw_img, bboxes = visualizer.draw_labeled_bboxes(draw_img,labels)
    return draw_img, bboxes

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