class NasParam:
    '''NAS for parameter searching'''
    def __init__(self, model_class, param):
        self.model_class = model_class
        self.param = param

    def gen_param_set(self):
        self.param_set = []

    def search(self, train_x, train_y, val_x, val_y):

        metrics = []
        losses = []
        param_cache = []
        for _ in range(4):
            for param in self.param_set:
                model = self.model_class(param)

                '''train k-epochs to store the best metric / loss'''
                hist = model.train(train_x, train_y, val_x, val_y)

                metrics.append(hist.metrics())
                losses.append(hist.losses())
                param_cache.append(param)

        '''find the best params'''

class NasConn:
    '''NAS for layer connection searching'''
    def __init__(self):
        pass


class NasRandom:
    '''NAS for random architecture searching'''
    def __init__(self):
        pass






