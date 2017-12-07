from bounding_box import BoundingBox

class Track(object):

    def __init__(self,box,id):
        self.box = box
        self.vel_x = 0
        self.vel_y = 0
        self.id = id
        self.age = 0

    def predict(self):
        self.box.x_center += self.vel_x
        self.box.y_center += self.vel_y

    def update(self,box):
        dx = box.x_center - self.box.x_center
        dy = box.y_center - self.box.y_center
        # print(self.id,dx,dy)
        self.box.x_center += dx / 2
        self.box.y_center += dy / 2
        self.box.width = (self.box.width + box.width) / 2
        self.box.height = (self.box.height + box.height) / 2
        self.vel_x = dx
        self.vel_y = dy
        self.age += 1