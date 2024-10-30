import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    expansion = 4  

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        # 1x1 Convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 Convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 Convolution
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Adding the skip connection
        out += identity
        out = self.relu(out)

        return out
    



class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet50, self).__init__()
        self.in_channels = 64

        # Initial Convolution and MaxPooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers (Each Layer is a series of Bottleneck Blocks)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)  # conv2_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # conv3_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # conv4_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # conv5_x

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, 512)  
        self.fc2 = nn.Linear(512, 256)                        
        self.fc3 = nn.Linear(256, num_classes)     
        self.sigmoid = nn.Sigmoid()     


    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        # If stride is not 1 or in_channels is not the same as out_channels*expansion, downsample
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        # Initial Layare
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Middle Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head Layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)  
        x = self.fc2(x)
        x = self.relu(x)  
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x
    
def resnet50():
    return ResNet50(BottleneckBlock, [3, 4, 6, 3])  


if __name__ == "__main__":
    model = resnet50()
    print(model)

    # Test
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size of 1, with 3 channels (RGB) and 224x224 image size
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Expected output shape: [1, 1000]

