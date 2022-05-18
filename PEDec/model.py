import torch.nn as nn
import torch.nn.functional as F
import torch
torch.manual_seed(100)

class Net_0D(nn.Module):
    def __init__(self,num_uniqFeature):
        super(Net_0D, self).__init__()

        self.embed = nn.Embedding(num_uniqFeature,528)
        self.conv1 = nn.Conv2d(1, 4, 20,stride = 20)
        self.BN1 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d((10,5),stride =(10,5))
        self.fc1 = nn.Linear(4 * 25*5, 64)
        self.fc2 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = torch.mul(self.embed.weight,x.reshape((x.shape[0],1,x.shape[2],1)).float())
        x = self.BN1(self.pool(F.relu(self.conv1(x))))
        x = x.view(-1, 4 * 25*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x




