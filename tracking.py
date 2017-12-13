from track import Track
import numpy as np
class Tracking(object):

    def __init__(self):
        self.list_of_tracks = []
        self.scaling_measurement = 0.3
        self.id_counter = 0
        self.threshold_bad_track = 0.85

    def get_number_of_tracks(self):
        return len(self.list_of_tracks)

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
                self.list_of_tracks[pair[1]].update(bbox,self.scaling_measurement)

        # Tracking management
        for track in self.list_of_tracks:
            if not track.has_been_updated:
                track.not_updated += 1
            track.belief = (track.age - track.not_updated)/track.age

        self.list_of_tracks = [x for x in self.list_of_tracks if x.belief > self.threshold_bad_track]

    def data_association(self,bboxes):

        association= []
        for i,box in enumerate(bboxes):
            min_distance = 100
            match = -1
            for j,track in enumerate(self.list_of_tracks):
                dx = (track.box.x_center - box.x_center)
                dy = (track.box.y_center - box.y_center)
                dw = (track.box.width - box.width)
                dh = (track.box.height - box.height)
                # distance = np.sqrt(dx ** 2 + dy ** 2)
                distance = np.sqrt(dx ** 2 + dy ** 2 + dw ** 2 + dh ** 2)
                if distance < min_distance:
                    min_distance = distance
                    match = j

            association.append([i,match])

        return association