# Imports
import visualizer
from classifier import Classifier
import os
import cv2
import skvideo.io
import imageio
# import numpy as np
import heatmap
from tracking import Tracking
from debugger import Debugger
from bounding_box import BoundingBox

# Hyperparameters
detect = True
test = True
image_size = (1280,720)
num_examples = 85
hm_threshold = 0
time_range = range(0,1260)
# time_range = range(600,700,10)


# Train classifier
if detect:
    test_image = cv2.imread('test_images/test6.jpg')
    svm = Classifier(num_examples,test_image)
    svm.train()

# See test images
if test:
    for image_name in os.listdir('test_images/'):

        # Dont use hidden files
        if image_name[0] == '.':
            continue

        # Read image and convert to RGB
        image = cv2.imread('test_images/'+image_name)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        classified_image, bounding_boxes = svm.debug_classify(image)
        # svm.find_cars(image)
        visualizer.draw_image(classified_image,'FirstDetection',save=False)

        # bboxes = heatmap.cluster_bounding_boxes(image,bounding_boxes,0)
        # detection_img = visualizer.draw_labeled_bboxes(image, bboxes)
        # visualizer.draw_image(detection_img,'FinalDetection',save=True)
        #
        # visualizer.draw_roi(image)

# See video
else:
    filename = 'project_video.mp4'

    vid = imageio.get_reader(filename,  'ffmpeg')
    debug = Debugger()

    if detect:
        for i in time_range:
            if i%100 ==0:
                print(i)
            image = vid.get_data(i)
            bboxes = svm.classify(image)
            bboxes = heatmap.cluster_bounding_boxes(image,bboxes,hm_threshold)

            # draw_img = visualizer.draw_labeled_bboxes(image, bboxes)
            # visualizer.draw_image(draw_img)

            debug.store_detected_bounding_boxes(bboxes,i)

        debug.write_detection()

    if not detect:
        detections = debug.read_detected_bounding_boxes()
        tracker = Tracking()

        writer = skvideo.io.FFmpegWriter("outputvideo.mp4")

        for i in range(len(detections)):
            frame = detections[i]['frame']
            image = vid.get_data(frame)
            boxes = detections[i]['boxes']
            bboxes = []
            for dict_box in boxes:
                bboxes.append(BoundingBox(dict_box))

            # detection_img = visualizer.draw_labeled_bboxes(image, bboxes)
            # visualizer.draw_image(detection_img,'FinalDetection' + str(i),save=False)
            # writer2.write(detection_img)


            tracker.prediction()

            tracker.update(bboxes)

            track_result = visualizer.draw_tracking(image, tracker)
            # visualizer.draw_image(track_result)
            writer.writeFrame(track_result)
            # print(i,tracker.get_number_of_tracks())
