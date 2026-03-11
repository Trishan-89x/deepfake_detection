import torch
import torch.nn as nn

class FrequencyCNN(nn.Module):

    def __init__(self):
        super(FrequencyCNN,self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.fc = nn.Sequential(

            nn.Linear(128*32*32,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,2)

        )

    def forward(self,x):

        x = self.conv(x)

        x = x.view(x.size(0),-1)

        x = self.fc(x)

        return x