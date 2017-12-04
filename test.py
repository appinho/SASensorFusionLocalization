import cv2
import visualizer
import color_histogram
import matplotlib.image as mpimg

image = mpimg.imread('test_images/test4.jpg')
rescaled_image = visualizer.rescale(image)
gray = cv2.cvtColor(rescaled_image,cv2.COLOR_RGB2GRAY)
# bboxes = [((800, 400), (950, 510)), ((1040, 390), (1260, 510))]
# bbox_image = visualizer.draw_boxes(image,bboxes)
visualizer.draw_image(image)

# rh, gh, bh, bincen, feature_vec = color_histogram.color_hist(image, nbins=32, bins_range=(0, 256))
# visualizer.draw_histograms(rh, gh, bh, bincen)

# visualizer.draw_image(rescaled_image)

# print(len(color_histogram.bin_spatial(image)))

# Define HOG parameters
orient = 9
pix_per_cell = 8
cell_per_block = 2
features, hog_image = color_histogram.get_hog_features(gray, orient,
                        pix_per_cell, cell_per_block,
                        vis=True, feature_vec=False)
visualizer.draw_image(hog_image)