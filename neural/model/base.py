import torch
import torch.nn as nn
class BaseModel(nn.Module):
    def __init__(self, param):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.param = param
        self.model = None

    def build(self):
        size_output = self.param['size_output']
        size_input = self.param['size_input']

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        # self.device.to(self.model)
        # self.tensor = self.tensor.to(self.device)





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
