
class BoundingBox(object):

    def __init__(self,box):
        self.x_center = (box[0][0]+box[1][0])/2
        self.y_center = (box[0][1]+box[1][1])/2
        self.width = box[1][0]-box[0][0]
        self.height = box[1][1]-box[0][1]
