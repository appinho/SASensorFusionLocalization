import visualizer
from classifier import Classifier
import os
import cv2
from skvideo.io import VideoWriter
import imageio
# import numpy as np
import heatmap
from tracking import Tracking
from debugger import Debugger
from bounding_box import BoundingBox

# Hyperparameters
write_detection = True
image_size = (1280,720)
num_examples = 8792
hm_threshold = 1

# Train classifier
if write_detection:
    svm = Classifier(num_examples,image_size)
    svm.train()

# See test images
# for image_name in os.listdir('test_images/'):
#
#     # Dont use hidden files
#     if image_name[0] == '.':
#         continue
#
#     # Read image and convert to RGB
#     image = cv2.imread('test_images/'+image_name)
#     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#
#     classified_image, bounding_boxes = svm.classify(image)
#     visualizer.draw_image(classified_image,'FirstDetection',save=True)
#
#     bboxes = heatmap.cluster_bounding_boxes(image,bounding_boxes,0)
#     detection_img = visualizer.draw_labeled_bboxes(image, bboxes)
#     visualizer.draw_image(detection_img,'FinalDetection',save=True)
#
#     visualizer.draw_roi(image)

# See video
filename = 'project_video.mp4'

vid = imageio.get_reader(filename,  'ffmpeg')
debug = Debugger()
time_range = range(0,1260)

if write_detection:
    for i in time_range:
        if i%100 ==0:
            print(i)
        image = vid.get_data(i)
        result1,bboxes = svm.classify(image)
        # visualizer.draw_image(result1)
        bboxes = heatmap.cluster_bounding_boxes(image,bboxes,hm_threshold)

        draw_img = visualizer.draw_labeled_bboxes(image, bboxes)
        # visualizer.draw_image(draw_img)

        debug.store_detected_bounding_boxes(bboxes,i)

    debug.write_detection()

if not write_detection:
    detections = debug.read_detected_bounding_boxes()
    tracker = Tracking()

    writer1 = VideoWriter('tracking_res.mp4',frameSize=image_size,fps=24)
    writer1.open()

    # writer2 = VideoWriter('detection_res.mp4',frameSize=image_size,fps=10)
    # writer2.open()

    for i in range(len(detections)):
        frame = detections[i]['frame']
        image = vid.get_data(frame)
        boxes = detections[i]['boxes']
        bboxes = []
        for dict_box in boxes:
            bboxes.append(BoundingBox(dict_box))

        # detection_img = visualizer.draw_labeled_bboxes(image, bboxes)
        # # visualizer.draw_image(detection_img,'FinalDetection',save=True)
        # writer2.write(detection_img)


        tracker.prediction()

        tracker.update(bboxes)

        track_result = visualizer.draw_tracking(image, tracker)
        writer1.write(track_result)