import numpy as np
import cv2
from skimage.feature import hog

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Returns the HOG features of an image
    :param img: Input image
    :param orient: Orientations
    :param pix_per_cell: Pixels per cell
    :param cell_per_block: Cells per block
    :param vis: Flag to include visualization
    :param feature_vec: Flag for feature vector creation
    :return:
    """

    if vis:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=True, feature_vector=False,
                                  block_norm="L2-Hys")
        return features, hog_image
    else:
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=False, feature_vector=feature_vec,
                                  block_norm="L2-Hys")
        return features

def bin_spatial(img, color_space='RGB', size=(32, 32)):
    """
    Converts image into a feature vector
    :param img: Input image
    :param color_space: Used color space of the image
    :param size: Size of the image
    :return: 1D Feature vector
    """
    # Convert image to new color space (if specified)
    if not color_space == 'RGB':
        if color_space == 'HLS':
            img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
        if color_space == 'HSV':
            img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img,size).ravel()

    # Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Returns the color histograms of an image as well as the bin centers and histogram features
    :param img: Input image
    :param nbins: Number of bins
    :param bins_range: Range of bins
    :return: color histograms, bin centers, histogram features
    """
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features