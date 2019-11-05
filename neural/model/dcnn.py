from neural.model import BaseModel
import torch.autograd
import torch
import torch.nn as nn
import torch.nn.functional as F

class DilateCnn(BaseModel):
    def build(self):
        self.conv1 = nn.Conv1d(1,8,3)
        self.conv2 = nn.Conv1d(8,16,3)
        self.fc = nn.Linear(16*16*16, 120)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc(x))

        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features





if __name__ == '__main__':
    param = {'batch_size':50}
    model = DilateCnn(param)
    model.build()
    print(model)
    p = list(model.parameters())
    print(len(p))
    print(p[0].size())
    input = torch.randn(1,1,32,32)
    out = model(input)
    print(out)

        

