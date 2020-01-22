# ReverseLayerF Class
from torch.autograd import Function

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


class DANN(nn.Module):
    """ Implementation of the DANN model """
    def __init__(self):
        super(DANN, self).__init__()
        self.cnn_layer = nn.Sequential()
        self.cnn_layer.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.cnn_layer.add_module('f_bn1', nn.BatchNorm2d(64))
        self.cnn_layer.add_module('f_pool1', nn.MaxPool2d(2))
        self.cnn_layer.add_module('f_relu1', nn.ReLU(True))
        self.cnn_layer.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.cnn_layer.add_module('f_bn2', nn.BatchNorm2d(50))
        self.cnn_layer.add_module('f_drop1', nn.Dropout2d())
        self.cnn_layer.add_module('f_pool2', nn.MaxPool2d(2))
        self.cnn_layer.add_module('f_relu2', nn.ReLU(True))

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
        output = self.cnn_layer(input_data)
        output = output.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(output, alpha)
        out_class = self.classify_class(output)
        out_domain = self.classify_domain(reverse_feature)

        return out_class, out_domain
