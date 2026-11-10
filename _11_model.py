'''
Visual - Plot training loss and validation loss per epoch
Model - start with basic model - Underfit
Size - 
                Input 64x64x3
                
                Conv1: 16 filters -> 64x64x16                   64x64x32
                Pool1: 32x32x16                                 32x32x32
                
                Conv2: 32 filters -> 32x32x32                   32x32x64
                Pool2: 16x16x32                                 16x16x64
                
                AvgPool1: 8x8x32                                8x8x64
                Flatten: 2048                                   4096
                FC1: Out features -> 28                   
'''
import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, n_outputs = 28):
        super().__init__()
        
        # ------------ FIRST LAYER ------------
        # in_channels = input channels (1 grayscale, 3 RGB), out_channels = number of filters, kernel_size = size of the filter, padding = amount of padding
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)
        # adds nonlinearity, nonnegative values, helps with vanishing gradient problem
        self.relu = nn.ReLU()
        # reduces spatial size of the input, helps with overfitting, reduces computation
        # stride is how much to move the filter 
        # halves the images by keeping the strongest value in a 2x2 block (2 is standard)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # ------------ SECOND LAYER ------------
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # shrinks each 16x16 map to 8x8 by averaging 2x2 blocks
        self.avgpool1 = nn.AdaptiveAvgPool2d(8)
        
        #self.dropout = nn.Dropout(p=0.3)
        
        # connected layer, inputs and outputs 
        # there are 32 channels of 8x8 images 
        self.fc1 = nn.Linear(in_features = 64 * 8 * 8, out_features = n_outputs)
        

    def forward(self, x):   
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.avgpool1(x)
        # flatten the tensor to feed into the fully connected layer
        # batch size, channels, height, width --> batch size, channels * height * width
        x = x.flatten(1)
        # x = self.dropout(x)
        x = self.fc1(x)
        return x
   
