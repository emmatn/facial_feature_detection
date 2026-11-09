'''
Visual - Plot training loss and validation loss per epoch
Model - start with basic model - Underfit
Size - 
                Input 64x64x3
                
                Conv1: 16 filters -> 64x64x16
                Pool1: 32x32x16
                AvgPool1: 4x4x16
                Flatten: 16,384 -> now 256 bc of adaptive pooling to 4x4
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
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1)
        # adds nonlinearity, nonnegative values, helps with vanishing gradient problem
        self.relu = nn.ReLU()
        # reduces spatial size of the input, helps with overfitting, reduces computation
        # stride is how much to move the filter 
        # halves the images by keeping the strongest value in a 2x2 block (2 is standard)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # shrinks each 32×32 map to 4×4 by averaging 8×8 blocks
        
        self.avgpool1 = nn.AdaptiveAvgPool2d(4)
        # connected layer, inputs and outputs 
        # images begin at 64x64, after conv1 and pool, they become 32x32, with 16 channels, so the input to the fully connected layer is 16 x 32 x 32
        # 14 landmarks, each with x and y coordinates, so the output of the fully connected layer is 14 x 2 = 28
        self.fc1 = nn.Linear(in_features = 16 * 4 * 4, out_features = n_outputs)
        
        # ------------ SECOND LAYER ------------
        # self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.avgpool1(x)
        # flatten the tensor to feed into the fully connected layer
        # batch size, channels, height, width --> batch size, channels * height * width
        x = x.flatten(1)
        x = self.fc1(x)
        return x
   
