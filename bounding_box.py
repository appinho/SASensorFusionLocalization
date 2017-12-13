
class BoundingBox(object):

    def __init__(self,info):
        self.x_center = info[0]
        self.y_center = info[1]
        self.width = float(info[2])
        self.height = float(info[3])
        self.p1 = (int(info[0] - info[2] / 2), int(info[1] - info[3] / 2))
        self.p2 = (int(info[0] + info[2] / 2), int(info[1] + info[3] / 2))

    def get_array(self):
        return [self.x_center,self.y_center,
                self.width,self.height]