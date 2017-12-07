import matplotlib.image as mpimg
import visualizer
from classifier import Classifier
import os
import cv2
from skvideo.io import VideoWriter
import imageio
import numpy as np
import speed_detection
import heatmap
from tracking import Tracking

image_size = (1280,720)
num_examples = 100
hm_threshold = 0

# Train classifier
svm = Classifier(num_examples)
svm.train()

# See test images
# for image_name in os.listdir('test_images/'):
#     if image_name[0] == '.':
#         continue
#     image = cv2.imread('test_images/'+image_name)
#     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#
#     classified_image, bounding_boxes = svm.classify(image)
#     visualizer.draw_image(classified_image)
#
#     cluster_img = heatmap.cluster_bounding_boxes(image,bounding_boxes,0)
#     visualizer.draw_image(cluster_img)

    # visualizer.draw_image(classified_image)



    # speed_image = svm.find_cars(image)
    #
    # visualizer.draw_image(speed_image)
# See video

filename = 'project_video.mp4'

vid = imageio.get_reader(filename,  'ffmpeg')
#
writer1 = VideoWriter('tracking_video.mp4',frameSize=image_size,fps=20)
writer1.open()
# writer2 = VideoWriter('output_video_fast.mp4',frameSize=image_size,fps=20)
# writer2.open()

tracker = Tracking()

for i in range(600,620):
    tracker.prediction()
    image = vid.get_data(i)
    visualizer.draw_tracking(image,tracker,[255,255,0])
    # print(np.max(image))
    result1,bboxes = svm.classify(image)
    # visualizer.draw_image(result1)
    cluster_img,cluster_bboxes = heatmap.cluster_bounding_boxes(image,bboxes,hm_threshold)
    # visualizer.draw_image(cluster_img)
    # writer1.write(result1)
    # result2 = svm.find_cars(image)
    # writer2.write(result2)
    tracker.update(cluster_bboxes)
    track_result = visualizer.draw_tracking(image,tracker,[0,255,0])
    writer1.write(track_result)
