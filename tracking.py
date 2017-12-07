from track import Track
import numpy as np
class Tracking(object):

    def __init__(self):
        self.list_of_tracks = []
        self.id_counter = 0

    def prediction(self):
        for track in self.list_of_tracks:
            track.predict()

    def update(self,bboxes):
        association = self.data_association(bboxes)
        # print(association)
        for pair in association:
            bbox = bboxes[pair[0]]

            # New track
            if pair[1] is -1:
                self.id_counter += 1
                new_track = Track(bbox,self.id_counter)
                self.list_of_tracks.append(new_track)
            else:
                self.list_of_tracks[pair[1]].update(bbox)


    def data_association(self,bboxes):

        association= []
        for i,box in enumerate(bboxes):
            min_distance = 2000
            match = -1
            for j,track in enumerate(self.list_of_tracks):
                dx2 = (track.box.x_center - box.x_center) ** 2
                dy2 = (track.box.y_center - box.y_center) ** 2
                distance = np.sqrt(dx2+dy2)
                if distance < min_distance:
                    min_distance = distance
                    match = j

            association.append([i,match])

        return association
