import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np



class RNN_P2(nn.Module):
    def __init__(self, input_size, n_layers=2):
        super(RNN_P2, self).__init__()
        self.lstm = nn.LSTM(input_size, 2048, n_layers, bidirectional=False)
        self.bn_0 = nn.BatchNorm1d(2048)
        self.fc_1 = nn.Linear(2048, 1024)
        #self.relu_1 = nn.ReLU()
        self.bn_1 = nn.BatchNorm1d(1024)
        self.fc_2 = nn.Linear(1024, 512)
        #self.relu_2 = nn.ReLU()
        self.bn_2 = nn.BatchNorm1d(512)
        self.fc_3 = nn.Linear(512, 11)
        self.softmax = nn.Softmax(1)
        
    def forward(self, input_sequence, seq_length, hidden_seq=None):
        input_seq = torch.nn.utils.rnn.pack_padded_sequence(input_sequence, seq_length)
        outputs, (hidden_unit, _) = self.lstm(input_seq, hidden_seq)  
        hidden_output = hidden_unit[-1]
        outputs = self.bn_0(hidden_output)
        outputs = self.fc_1(outputs)
        outputs = self.bn_1(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.bn_2(outputs)
        outputs = self.softmax(self.fc_3(outputs))
        return outputs, hidden_output