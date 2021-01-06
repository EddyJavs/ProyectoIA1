class Data:
    def __init__(self, data_set_x, data_set_y, max_value=1):
        self.m = data_set_x.shape[1]
        self.n = data_set_x.shape[0]
        self.x = data_set_x / max_value #escalacion de variables
        self.y = data_set_y
