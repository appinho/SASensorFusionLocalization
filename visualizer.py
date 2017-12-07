import numpy as np
import cv2
import matplotlib.pyplot as plt
from bounding_box import BoundingBox

def draw_spatially_binned_features(feature_vec):
    """
    Draws spatially binned feature vector
    :param feature_vec: Feature vector
    :return: None
    """

    # Draw feature vector
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')

def rescale(image):
    """
    Rescales image on 32x32 size
    :param image: Input image
    :return: Returns 32x32 pixel version of image
    """

    # Return resized image
    return cv2.resize(image,(32,32))

def draw_histograms(r_hist, g_hist, b_hist, bin_centers):
    """
    Draws the histograms of the RGB channels
    :param r_hist: Red Channel Histogram
    :param g_hist: Grenn Channel Histogram
    :param b_hist: Blue Channel Histogram
    :param bin_centers: Bin centers
    :return: None
    """

    # Draw histograms
    fig = plt.figure(figsize=(12,3))
    plt.subplot(131)
    plt.bar(bin_centers, r_hist[0])
    plt.xlim(0, 256)
    plt.title('R Histogram')
    plt.subplot(132)
    plt.bar(bin_centers, g_hist[0])
    plt.xlim(0, 256)
    plt.title('G Histogram')
    plt.subplot(133)
    plt.bar(bin_centers, b_hist[0])
    plt.xlim(0, 256)
    plt.title('B Histogram')
    fig.tight_layout()
    plt.show()

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Returns a copy of the image with drawn bounding boxes
    :param img: Image without bounding box information
    :param bboxes: Bounding boxes
    :param color: Color code of the bounding boxes
    :param thick: Thickness of lines from the bounding boxes
    :return: Image with bounding box information
    """

    # Make a copy of the image
    draw_img = np.copy(img)

    # Loop through bounding boxes and add them
    for bbox in bboxes:
        # print(bbox)
        cv2.rectangle(draw_img,bbox[0],bbox[1],color=color,thickness=thick)

    # Returns image with bounding boxes
    return draw_img

def draw_image(img):
    """
    Draws image
    :param img: Image to draw
    :return: None
    """

    # Draw image
    plt.imshow(img, cmap='gray')
    plt.show()

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(BoundingBox(bbox))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img,bboxes

def draw_tracking(image,tracker,color):
    draw_img = np.copy(image)

    for track in tracker.list_of_tracks:
        p1_x = np.int(track.box.x_center - track.box.width/2)
        p2_x = np.int(track.box.x_center + track.box.width / 2)
        p1_y = np.int(track.box.y_center - track.box.height / 2)
        p2_y = np.int(track.box.y_center + track.box.height / 2)
        cv2.rectangle(draw_img, (p1_x,p2_y), (p2_x,p1_y), color=color, thickness=3)

    # draw_image(draw_img)
    return draw_img