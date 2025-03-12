"""
Code to manipulate the signal dataset optimized for pytorch.
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

from dataclasses import dataclass, field
from typing import Any
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import utils.config as config
from utils.helpers import get_logger

logger = get_logger(__file__)


@dataclass(kw_only=True)
class InfrasoundDataset(Dataset):
    """
    Class to create Infrasound dataset from audio file + annotation file
    The Class applies the following functions
    1. A Transformation specified in the configuration file, This can either be MFCC, Mel-Spectrgram or any added
        in the future
    2. A function to mix down audio to mono channel if more than on channel
    3. A function to resample data to target sample rate specified in the config file
    4. A function to pad the audio if sample is less that target sample
    5. A function to trim sample

    Args:
        labels_file: path to file containing labels
        waveform_file: path to file containing audio signals
        transformation: features to be extracted from signal
        target_sample_rate: the sample rate, defaults to 100Mhz
        num_samples: size of chunks, each sec contains 100 samples
        device: str: either of CPU or GPU

    Returns:
        signal: processed audio signal
        label: resulting label
    """

    labels_file: str
    waveform_file: str
    transformation: Any
    target_sample_rate: int
    num_samples: int
    device: str = config.DEVICE
    labels: Any = field(init=False)
    waveform_raw: Any = field(init=False)

    def __post_init__(self) -> None:
        self.labels = pd.read_parquet(self.labels_file)
        self.waveform_raw = pd.read_parquet(self.waveform_file)
        self.transformation = self.transformation.to(self.device)
        logger.info(f"Device in use: {self.device}")

    def __len__(self):
        """
        Returns: The len of processed data
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Function to process every input
        Args:
            index: index of input to be processed
        Returns:
            audio feature corresponding to the applied transformation
        """
        label = self._get_audio_sample_label(index)
        signal = torch.tensor(
            np.array(self.waveform_raw.iloc[index]).reshape(1, 1000)
        ).float()
        _sr = config.SAMPLE_RATE
        signal = signal.to(self.device)
        signal = self._resample_if_not_target_sample_rate(signal, _sr)
        signal = self._mix_down_if_not_mono(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        """
        Args:
            signal: Some audio signal
        Returns:
            truncated audio signal, to ensure all signals have the same len or num of samples
        """
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        """
        Pads signal if the length is shorter that the specified number of sample in the config file
        Args:
            signal: Some audio signal
        Returns: Right padded audio
        """
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_not_target_sample_rate(self, signal, sr):
        """
        Function to resample audio to sample target rate
        Args:
            signal: Some audio signal
            sr: Sample rate from config file
        Returns: resampled audio data
        """
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(
                self.device
            )
            signal = resampler(signal)
        return signal

    @staticmethod
    def _mix_down_if_not_mono(signal):
        """
        Function to reduce audio to one/mono channel
        Args:
            signal: some audion data
        Returns: mono channel audio
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_label(self, index):
        return self.labels.iloc[index, 0]


if __name__ == "__main__":

    # isd = InfrasoundDataset(
    #     labels_file=config.THIS_LABEL_FILE,
    #     waveform_file=config.THIS_WAVEFORM_FILE,
    #     transformation=config.TRANSFORMATION,
    #     target_sample_rate=config.SAMPLE_RATE,
    #     num_samples=config.NUM_SAMPLES,
    # )
    #
    # test_isd = InfrasoundDataset(
    #     labels_file=config.THIS_LABEL_FILE_TEST,
    #     waveform_file=config.THIS_WAVEFORM_FILE_TEST,
    #     transformation=config.TRANSFORMATION,
    #     target_sample_rate=config.SAMPLE_RATE,
    #     num_samples=config.NUM_SAMPLES,
    # )

    isd = InfrasoundDataset(
        labels_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_y_train_.parquet",
        waveform_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_X_train_.parquet",
        transformation=config.TRANSFORMATION,
        target_sample_rate=config.SAMPLE_RATE,
        num_samples=config.NUM_SAMPLES,
    )

    test_isd = InfrasoundDataset(
        labels_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_y_test_.parquet",
        waveform_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_X_test_.parquet",
        transformation=config.TRANSFORMATION,
        target_sample_rate=config.SAMPLE_RATE,
        num_samples=config.NUM_SAMPLES,
    )


    logger.info(f"There are {len(isd)} samples in the dataset.")
    logger.info(f"There are {len(test_isd)} samples in the test dataset.")
    this_signal, this_label = isd[1]

    logger.info(this_signal.size())

