""" 
    This module contains improved UDA model
    It is the modified version of DANN model uses Resnet34 as the backed
"""
# ReverseLayerF Class
from torch.autograd import Function
import torchvision.models as models

# DANN Classs
import torch.nn as nn
 


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANN_IMPROVED(nn.Module):
    """ Implementation of the DANN model """
    def __init__(self):
        super(DANN_IMPROVED, self).__init__()

        # The model feature extractor use resnet34 as backend
        self.resnet34 = models.resnet34(pretrained = True)
        self.resnet34 = nn.Sequential(*(list(self.resnet34.children())[:-2]))
        for params in self.resnet34.parameters():   
            params.requires_grad = False
        self.resnet34.add_module('r_convt1', nn.ConvTranspose2d(in_channels=512,
                                         out_channels=256, 
                                         kernel_size=4, 
                                         stride=2, padding=1, bias=False))
        self.resnet34.add_module('r_relu1', nn.ReLU(inplace=True))
        self.resnet34.add_module('r_convt2', nn.ConvTranspose2d(in_channels=256,
                                 out_channels=128, 
                                 kernel_size=4, 
                                 stride=2, padding=1, bias=False))
        self.resnet34.add_module('r_relu2', nn.ReLU(inplace=True))
        self.resnet34.add_module('r_convt3', nn.ConvTranspose2d(in_channels=128,
                                 out_channels=64, 
                                 kernel_size=4, 
                                 stride=2, padding=1, bias=False))
        self.resnet34.add_module('r_relu3', nn.ReLU(inplace=True))
        self.resnet34.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.resnet34.add_module('f_bn2', nn.BatchNorm2d(50))
        self.resnet34.add_module('f_drop1', nn.Dropout2d())
        self.resnet34.add_module('f_relu2', nn.ReLU(True))


        # This part of the cnn acts as an class classifier
        self.classify_class = nn.Sequential()
        self.classify_class.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.classify_class.add_module('c_bn1', nn.BatchNorm1d(100))
        self.classify_class.add_module('c_relu1', nn.ReLU(True))
        self.classify_class.add_module('c_drop1', nn.Dropout2d())
        self.classify_class.add_module('c_fc2', nn.Linear(100, 100))
        self.classify_class.add_module('c_bn2', nn.BatchNorm1d(100))
        self.classify_class.add_module('c_relu2', nn.ReLU(True))
        self.classify_class.add_module('c_fc3', nn.Linear(100, 10))
        self.classify_class.add_module('c_softmax', nn.LogSoftmax(dim=1))

        # This part of the CNN acts as a domain classifier
        self.classify_domain = nn.Sequential()
        self.classify_domain.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.classify_domain.add_module('d_bn1', nn.BatchNorm1d(100))
        self.classify_domain.add_module('d_relu1', nn.ReLU(True))
        self.classify_domain.add_module('d_fc2', nn.Linear(100, 2))
        self.classify_domain.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        output = self.resnet34(input_data)
        output = output.view(-1, 50 * 4 * 4)

        reverse_feature = ReverseLayerF.apply(output, alpha)
        out_class = self.classify_class(output)
        out_domain = self.classify_domain(reverse_feature)

        return out_class, out_domain
