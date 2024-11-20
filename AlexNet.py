from torch import nn,flatten
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AlexNet,self).__init__(*args, **kwargs)
        self.conv1=nn.Conv2d(3,96,(11,11),4,0)
        self.max_pool1=nn.MaxPool2d((3,3),2)
        self.conv2=nn.Conv2d(96,256,(5,5),1,2)
        self.max_pool2=nn.MaxPool2d((3,3),2)
        self.conv3=nn.Conv2d(256,384,(3,3),1,1)
        self.conv4=nn.Conv2d(384,384,(3,3),1,1)
        self.conv5=nn.Conv2d(384,256,(3,3),1,1)
        self.max_pool3=nn.MaxPool2d((3,3),2)
        self.fc1=nn.Linear(9216,4096)
        self.dropout1=nn.Dropout()
        self.fc2=nn.Linear(4096,4096)
        self.dropout2=nn.Dropout()
        self.fc3=nn.Linear(4096,10)
        
        
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.max_pool1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.max_pool2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=F.relu(self.conv5(x))
        x=F.relu(self.max_pool3(x))
        x=flatten(x)
        x=F.relu(self.fc1(x))
        x=F.relu(self.dropout1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.dropout2(x))
        x=F.relu(self.fc3(x))
        return x
    
