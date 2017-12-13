import json

class Debugger(object):

    def __init__(self):
        self.detections = []
        self.filename = 'detection.json'

    def store_detected_bounding_boxes(self,boxes,frame):

        detection = dict()
        detection['frame'] = frame
        detection['boxes'] = []

        for box in boxes:
            detection['boxes'].append(box.get_array())

        self.detections.append(detection)

    def write_detection(self):
        with open(self.filename, 'w') as f:
            json.dump(self.detections, f)


        # with open(filename, "a") as file:
        #     for frame in boxes:
        #         for box in frame:
        #             file.write('%d' % box.x_center)
        #             file.write(',')
        #             file.write('%d' % box.y_center)
        #             file.write(',')
        #             file.write('%d' % box.height)
        #             file.write(',')
        #             file.write('%d' % box.width)
        #             file.write(' ')
        #         file.write('\n')

    def read_detected_bounding_boxes(self):
        with open(self.filename, 'r') as g:
            mydic_restored = json.loads(g.read())

        return mydic_restored
