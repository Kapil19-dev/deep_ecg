class BaseModel:
    def __init__(self, param):
        self.param = param

    def build(self):
        size_output = self.param['size_output']
        size_input = self.param['size_input']
        pass

    def train(self, train_x, train_y, val_x, val_y):
        pass

    def train_gen(self, train_iter, val_iter):
        pass


class BasePreprocessor:
    def __init__(self, param):
        self.param = param

    def label_enhance(self, label):
        pass

    def label_filter(self, label):
        pass

    def label_transform(self, label):
        pass

    def sig_enhance(self, sig):
        pass

    def sig_filter(self, sig):
        pass

    def sig_transform(self, sig):
        pass
