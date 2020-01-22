import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

"""
    Contains CNN feature extractor  
"""

class CNN_P1(nn.Module):
    """ Implementation of the CNN Feature Extractor model """
    def __init__(self, feature_size):
        super(CNN_P1, self).__init__()
        self.feature_size = feature_size
        # The model feature extractor use resnet50 as feature extractor
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50 = nn.Sequential(*(list(self.resnet50.children())[:-2]))
        # freeze parameters
        for params in self.resnet50.parameters():   
            params.requires_grad = False

 
        # This is the upper feature extractor part 
        last_feature = 2048
        self.classifier = nn.Sequential()
        self.classifier.add_module('c_l1', nn.Linear(last_feature,1024))
        self.classifier.add_module('c_bn1', nn.BatchNorm1d(1024))
        self.classifier.add_module('c_relu1', nn.ReLU(True))
        self.classifier.add_module('c_l2', nn.Linear(1024, 11))
        self.classifier.add_module('c_s1', nn.Softmax(1))
        


    def forward(self, input_data):

        input_data = input_data.expand(input_data.data.shape[0], 3, 64, 64)
        output = self.resnet50(input_data)

        out_class = self.classifier(output)

        return out_class 



"""
    Upper Extractor Seperate from the CNN_P1 model
"""
class CNN_P1_UPPER(torch.nn.Module):
    def __init__(self, out_cnn_p1):
        super(CNN_P1_UPPER, self).__init__()
        
        self.linear1 = nn.Linear(out_cnn_p1,1024)
        self.bn_1 = nn.BatchNorm1d(1024)
        self.linear2 = nn.Linear(1024, 11)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(1)
        
    def forward(self, x):
        layer_1 = self.relu(self.bn_1(self.linear1(x)))  
        pred = self.softmax(self.linear2(layer_1))
        return pred


