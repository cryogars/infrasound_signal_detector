"""
Training Engine
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

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from utils.infrasounddataset import InfrasoundDataset
from utils import config as config
from utils.helpers import get_logger
from cnn import CNNNetwork, CNNNetworkTemporal, CNNNetworkMeteorological

logger = get_logger(__file__)


def create_data_loader(train_data, batch_size):
    """
    Args:
        train_data: The train data
        batch_size: training batch size
    Returns: batched data for the training
    """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader


# Code to train a single epoch | Specify loss function | Specify Optimizer  | Specify device
def train_single_epoch(model, data_loader, _loss_fn, _optimizer, _device=config.DEVICE):
    """
    Args:
        model: Constructed Neural network
        data_loader: callable to load data on the model
        _loss_fn: Loss function for model
        _optimizer: Any of Adam, RMSProp, SGD to optimizer
        _device: CPU or GPU Device to process/train model

    Returns: None, trains single epochs
    """
    loss = None
    model.train()
    for _input, _target in data_loader:
        _input, _target = _input.to(_device), _target.to(_device)

        # compute loss
        prediction = model(_input)
        loss = _loss_fn(prediction, _target)

        # reset the gradient: backpropagation of error and update weights
        _optimizer.zero_grad()
        loss.backward()
        _optimizer.step()
    logger.info(f"Loss: {loss.item()}")
    return loss.item()


def train(model, data_loader, _loss_fn, _optimizer, epochs, model_save_name):
    """
    Args:
        model: The model
        data_loader: The data loadr object
        _loss_fn: The loss function
        _optimizer: The optimizer "SGD", "Adams", "RmProps"
        epochs: Number of Epochs
        model_save_name: ...

    Returns: None,
    """
    best_val_loss = np.inf
    for i in range(epochs):
        logger.info(f"Epoch {i + 1}")
        this_loss = train_single_epoch(
            model, data_loader, _loss_fn, _optimizer, _device=config.DEVICE
        )
        logger.info("---------------------------")
        if this_loss < best_val_loss:
            best_val_loss = this_loss
            # save model
            torch.save(
                model.state_dict(),
                f=f"{config.MODELS_FOLDER}/{model_save_name}.pth",
            )
            logger.info(
                f"Trained CNN saved at {model_save_name}.pth"
            )
    logger.info("Finished training")


if __name__ == "__main__":
    ############################################################################
    ## SENSOR SENSITIVE MODEL L3S3
    ############################################################################
    logger.info(f"Training Sensor Agnostic Sensor Case")

    temporal_isd = InfrasoundDataset(
        labels_file=f"{config.PROCESSED_DATA}/two_high_signal_l3s3/high_signal_y_train_.parquet",
        waveform_file=f"{config.PROCESSED_DATA}/two_high_signal_l3s3/high_signal_X_train_.parquet",
        transformation=config.TRANSFORMATION,
        target_sample_rate=config.SAMPLE_RATE,
        num_samples=config.NUM_SAMPLES,
    )
    train_dataloader = create_data_loader(temporal_isd, config.BATCH_SIZE)

    sensor_model = CNNNetwork().to(config.DEVICE)
    logger.info(sensor_model)
    loss_fn = nn.CrossEntropyLoss()  # initialise loss function
    sensor_optimizer = torch.optim.Adam(
        sensor_model.parameters(), lr=config.SENSOR_LEARNING_RATE
    )  # initialise optimizer

    # train model
    train(
        model=sensor_model,
        data_loader=train_dataloader,
        _loss_fn=loss_fn,
        _optimizer=sensor_optimizer,
        epochs=config.EPOCHS,
        model_save_name="sensor_model_l3_s3"
    )

    # ############################################################################
    # ## SENSOR SENSITIVE MODEL
    # ############################################################################
    # logger.info(f"Training Sensor Agnostic Sensor Case")
    #
    # temporal_isd = InfrasoundDataset(
    #     labels_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_y_train_.parquet",
    #     waveform_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_X_train_.parquet",
    #     transformation=config.TRANSFORMATION,
    #     target_sample_rate=config.SAMPLE_RATE,
    #     num_samples=config.NUM_SAMPLES,
    # )
    # train_dataloader = create_data_loader(temporal_isd, config.BATCH_SIZE)
    #
    # sensor_model = CNNNetwork().to(config.DEVICE)
    # logger.info(sensor_model)
    # loss_fn = nn.CrossEntropyLoss()  # initialise loss function
    # sensor_optimizer = torch.optim.Adam(
    #     sensor_model.parameters(), lr=config.SENSOR_LEARNING_RATE
    # )  # initialise optimizer
    #
    # # train model
    # train(
    #     model=sensor_model,
    #     data_loader=train_dataloader,
    #     _loss_fn=loss_fn,
    #     _optimizer=sensor_optimizer,
    #     epochs=config.EPOCHS,
    #     model_save_name="sensor_model"
    # )

    ############################################################################
    ## TEMPORAL SENSITIVE MODEL
    ############################################################################
    # logger.info(f"Training Temporal Sensitive Sensor Case")
    #
    # temporal_isd = InfrasoundDataset(
    #     labels_file=f"{config.PARENT_DIRECTORY}/data/processed/first_ten/y_train.parquet",
    #     waveform_file=f"{config.PARENT_DIRECTORY}/data/processed/first_ten/x_train.parquet",
    #     transformation=config.TRANSFORMATION,
    #     target_sample_rate=config.SAMPLE_RATE,
    #     num_samples=config.NUM_SAMPLES,
    # )
    # train_dataloader = create_data_loader(temporal_isd, config.BATCH_SIZE)
    #
    # temp_model = CNNNetworkTemporal().to(config.DEVICE)
    # logger.info(temp_model)
    # loss_fn = nn.CrossEntropyLoss()  # initialise loss function
    # temp_optimizer = torch.optim.Adam(
    #     temp_model.parameters(), lr=config.TEMP_LEARNING_RATE
    # )  # initialise optimizer
    #
    # # train model
    # train(
    #     model=temp_model,
    #     data_loader=train_dataloader,
    #     _loss_fn=loss_fn,
    #     _optimizer=temp_optimizer,
    #     epochs=config.EPOCHS,
    #     model_save_name="temporal_model"
    # )

    ############################################################################
    ## METEOROLOGICAL SENSITIVE MODEL
    ############################################################################

    # logger.info(f"Training Meteorological Sensitive  Case")
    # meteorological_isd = InfrasoundDataset(
    #     labels_file=f"{config.PARENT_DIRECTORY}/data/processed/even_odd/y_train.parquet",
    #     waveform_file=f"{config.PARENT_DIRECTORY}/data/processed/even_odd/x_train.parquet",
    #     transformation=config.TRANSFORMATION,
    #     target_sample_rate=config.SAMPLE_RATE,
    #     num_samples=config.NUM_SAMPLES,
    # )
    # train_dataloader = create_data_loader(meteorological_isd, config.BATCH_SIZE)
    #
    # met_model = CNNNetworkMeteorological().to(config.DEVICE)
    # logger.info(met_model)
    # loss_fn = nn.CrossEntropyLoss()  # initialise loss function
    # met_optimizer = torch.optim.Adam(
    #     met_model.parameters(), lr=config.MET_LEARNING_RATE
    # )  # initialise optimizer
    #
    # # train model
    # train(
    #     model=met_model,
    #     data_loader=train_dataloader,
    #     _loss_fn=loss_fn,
    #     _optimizer=met_optimizer,
    #     epochs=config.EPOCHS,
    #     model_save_name="meteorological_model"
    # )
