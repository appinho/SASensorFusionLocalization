class Track(object):

    def __init__(self,box,id):
        self.box = box
        self.vel_x = 0
        self.vel_y = 0
        self.id = id
        self.age = 1
        self.not_updated = 0
        self.has_been_updated = True
        self.belief = 0

    def predict(self):
        self.box.x_center += self.vel_x
        self.box.y_center += self.vel_y
        self.has_been_updated = False

    def update(self,box,scaling):
        dx = box.x_center - self.box.x_center
        dy = box.y_center - self.box.y_center
        dw = box.width - self.box.width
        dh = box.height - self.box.height
        # print(self.id,dx,dy)
        self.box.x_center += dx * scaling
        self.box.y_center += dy * scaling
        self.box.width += dw * scaling
        self.box.height += dh * scaling
        self.vel_x = dx * 0.1
        self.vel_y = dy * 0.1
        self.age += 1
        self.has_been_updated = True