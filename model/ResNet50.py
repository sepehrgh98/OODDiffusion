import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet50, self).__init__()
        # Load the pretrained ResNet50 model from torchvision
        pretrained_resnet = models.resnet50(pretrained=True)
        
        # Extract all layers except the final fully connected layer
        # We will keep up to the avgpool layer
        self.features = nn.Sequential(*list(pretrained_resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Keep the average pooling layer
        
        # Head
        self.fc1 = nn.Linear(512 * 4, 512)  # 512 * block.expansion, where expansion for Bottleneck is 4
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Pass the input through the feature extractor layers
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the output for feature vector
        
        # Pass through the head
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        # x = x.squeeze()
        
        return x

if __name__ == "__main__":
    # Instantiate the feature extractor
    feature_extractor = ResNet50()
    
    # Print the architecture of the feature extractor
    print(feature_extractor)

    # Test the feature extractor
    input_tensor = torch.randn(4, 3, 224, 224)  # Batch size of 1, with 3 channels (RGB) and 224x224 image size
    output = feature_extractor(input_tensor)
    print("Output shape (feature vector):", output.shape)  # Expected output shape: [1, num_classes]
