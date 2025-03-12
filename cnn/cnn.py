"""
CNN Architecture
"""

__author__ = "Evi Ofekeze"
__authors__ = ["HP Marshal", "Jefferey B Johnson"]
__contact__ = "eviofekeze@u.boisestate.edu"
__copyright__ = "Copyright 2024, Boise State University, Boise ID"
__group__ = "Cryosphere GeoPhysics and Remote Sensing Laboratory"
__credits__ = ["Evi Ofekeze", "HP Marshal", "Jefferey B Johnson"]
__email__ = "eviofekeze@u.boisestate.edu"
__maintainer__ = "developer"
__status__ = "Research"

import sys
sys.path.append('../')

from torch import nn
from torchsummary import summary
import utils.config as config
from utils.helpers import get_logger

logger = get_logger(__file__)


############################################################################
## SENSOR AGNOSTIC MODEL
############################################################################

class CNNNetwork(nn.Module):
    def __init__(self):
        """
        Constructor to initialize the CNN and add layers
        Add 4 Convolution Layers | Flattened Layer | Linear Layer | Softmax Layer
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # self.dropout = nn.Dropout(p=0.34)
        self.flatten = nn.Flatten()

        if config.TRANSFORMATION_NAME == "mel-spectrogram":
            linear_input_size = 128 * 3 * 3
        elif config.TRANSFORMATION_NAME == "mfcc":
            linear_input_size = 128 * 2 * 2

        self.linear = nn.Linear(
            in_features=linear_input_size, out_features=config.CLASSES
        )
        # self.dropout_fc1 = nn.Dropout(p=0.46)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        """
        Forward model to implement CNN for the input data
        Args:
            input_data:
        Returns:
        """
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.dropout(x)
        x = self.flatten(x)
        logit = self.linear(x)
        # logit = self.dropout_fc1(logit)
        predictions = self.logsoftmax(logit)
        return predictions



############################################################################
## METEOROLOGICAL SENSITIVE MODEL
############################################################################

class CNNNetworkMeteorological(nn.Module):

    def __init__(self):
        """
        Constructor to initialize the CNN and add layers
        Add 2 Convolution Layers | Flattened Layer | Linear Layer | Softmax Layer
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=80, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=80, out_channels=112, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.dropout = nn.Dropout(p=0.28)
        self.flatten = nn.Flatten()

        if config.TRANSFORMATION_NAME == "mel-spectrogram":
            linear_input_size = 112 * 9 * 9
        elif config.TRANSFORMATION_NAME == "mfcc":
            linear_input_size = 112 * 2 * 2

        self.linear = nn.Linear(
            in_features=linear_input_size, out_features=config.CLASSES
        )

        self.dropout_fc1 = nn.Dropout(p=0.26)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        """
        Forward model to implement CNN for the input data
        Args:
            input_data:
        Returns:
        """
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        logit = self.linear(x)
        logit = self.dropout_fc1(logit)
        predictions = self.logsoftmax(logit)
        return predictions


############################################################################
## TEMPORAL SENSITIVE MODEL
############################################################################

class CNNNetworkTemporal(nn.Module):

    def __init__(self):
        """
        Constructor to initialize the CNN and add layers
        Add Convolution Layers | Flattened Layer | Linear Layer | Softmax Layer
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=80, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=80, out_channels=96, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.dropout = nn.Dropout(p=0.4457)
        self.flatten = nn.Flatten()

        if config.TRANSFORMATION_NAME == "mel-spectrogram":
            linear_input_size = 96 * 9 * 9
        elif config.TRANSFORMATION_NAME == "mfcc":
            linear_input_size = 96 * 2 * 2

        self.linear = nn.Linear(
            in_features=linear_input_size, out_features=config.CLASSES
        )

        self.dropout_fc1 = nn.Dropout(p=0.4621)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        """
        Forward model to implement CNN for the input data
        Args:
            input_data:
        Returns:
        """
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        logit = self.linear(x)
        logit = self.dropout_fc1(logit)
        predictions = self.logsoftmax(logit)
        return predictions


############################################################################
## MAIN FUNCTION
############################################################################
if __name__ == "__main__":
    logger.debug(f"Sensor Agnostic Model")
    cnn = CNNNetwork()
    logger.info(f"Network Summary for Sensor Agnostic Model")
    if config.TRANSFORMATION_NAME == "mel-spectrogram":
        summary(cnn.cuda(), input_size=(1, 32, 32))
    elif config.TRANSFORMATION_NAME == "mfcc":
        summary(cnn.cuda(), input_size=(1, 13, 6))

    logger.debug(f"Temporal Sensitive Model")
    cnn_temp = CNNNetworkTemporal()
    logger.info(f"Network Summary for Temporal Sensitive Model")
    summary(cnn_temp.cuda(), input_size=(1, 32, 32))

    logger.debug(f"Meteorological Sensitive Model")
    cnn_met = CNNNetworkMeteorological()
    logger.info(f"Network Summary for Meteorological Sensitive Model")
    summary(cnn_met.cuda(), input_size=(1, 32, 32))
