import numpy as np
import cv2
import matplotlib.pyplot as plt
from bounding_box import BoundingBox

color = {
    0: [255, 0, 0],
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [255, 255, 0],
    4: [255, 0, 255],
    5: [0, 255, 255],
    6: [255, 0, 125],
    7: [255, 125, 0],
    8: [125, 255, 0],
    9: [0, 255, 125],
    10: [0, 125, 255],
    11: [125, 0, 255],
}

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

def draw_image(img,title = '',save=False):
    """
    Draws image
    :param img: Image to draw
    :return: None
    """

    # Draw image
    f = plt.gcf()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()
    if save:
        f.savefig('output_images/' + title + '.png')

def draw_two_images(img1,img2,title = '',save=False):

    # Draw image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img1, cmap='gray')
    ax2.imshow(img2, cmap='gray')
    plt.show()
    if save:
        f.savefig('output_images/' + title + '.png')

def draw_labeled_bboxes(img, bboxes):
    # Iterate through all detected cars
    for bbox in bboxes:
        # Draw the box on the image
        cv2.rectangle(img, bbox.p1, bbox.p2, (0,0,255), 6)
    # Return the image
    return img

def draw_tracking(image,tracker):
    draw_img = np.copy(image)

    for track in tracker.list_of_tracks:
        if track.age > 3:
            col = color[track.id % len(color)]
            p1_x = np.int(track.box.x_center - track.box.width/2)
            p2_x = np.int(track.box.x_center + track.box.width / 2)
            p1_y = np.int(track.box.y_center - track.box.height / 2)
            p2_y = np.int(track.box.y_center + track.box.height / 2)
            cv2.rectangle(draw_img, (p1_x,p2_y), (p2_x,p1_y), color=col, thickness=3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(draw_img, 'ID ' + str(track.id) + ' AGE ' + str(track.age),
                    (p1_x, p1_y - 10), font, 1, col, 2, cv2.LINE_AA)
            cv2.putText(draw_img, 'B ' + str('%.2f' % track.belief),
                    (p1_x, p1_y - 40), font, 1, col, 2, cv2.LINE_AA)
    # draw_image(draw_img)
    return draw_img

def read_and_draw_image(image_name,title):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    draw_image(image,title,True)
    return image
