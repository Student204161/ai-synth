import torch.nn.functional as F
import torch.nn

class MyImageClassifier(torch.nn.Module):
    """ Image classifier neural network class. 
    
    Args:
        in_channels: number of input channels in the image
        num_classes: number of classes for classification
    
    """
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(MyImageClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(128 * 7 * 7, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N, C, H, W]

        Returns:
            Output tensor with shape [N, num_classes]

        """
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
