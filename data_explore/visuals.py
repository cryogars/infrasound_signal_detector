"""
Script to prep sample data
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
sys.path.append("..")

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.feature

# User Defined libraries
import utils.config as config
from utils.helpers import get_logger

logger = get_logger(__name__)


def extract_mel_spectrogram_singular(signal):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        window="hann",
        hop_length=config.MEL_HOP_LENGTH,
        n_mels=config.N_MELS,
    )

    # y_axis = 'log'
    # win_length = 50,

    # mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db


def extract_mfcc_singular(signal):
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        n_mfcc=config.N_MFCC,
        hop_length=32,
        dct_type=config.DISCRETE_COSINE_TRANSFORM_TYPE,
        norm=config.NORM,
    )
    return mfcc

def frequency_spectrum(signal, spec: bool = False):
    window = np.hanning(len(signal))
    windowed_input = signal * window
    dft = np.fft.rfft(windowed_input)

    pa = np.abs(dft)
    pa_db = librosa.amplitude_to_db(pa, ref=np.max)
    frequency = librosa.fft_frequencies(n_fft=len(signal), sr=config.SAMPLE_RATE)

    # Spec computation
    if spec:
        D = librosa.stft(np.array(signal).astype(float), n_fft=config.N_FFT)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, x_axis="time", y_axis="log")
        plt.ylim(0,50)
        plt.colorbar()

    # plt.plot(frequency, pa_db)
    # plt.xscale("log")

    plt.show()


if __name__ == "__main__":
    sdf = pd.read_parquet(
        f"{config.PROCESSED_DATA}/two_high_signal/full_waveform_stats_df_LCC3_train_sensor_3.parquet"
    )

    signal_class = sdf.groupby("label_names").sample(n=1)
    signal_class.reset_index(inplace=True)
    logger.info(f"Signal DF")
    logger.info(signal_class)

    class_names = signal_class["label_names"].unique()
    logger.info(f"Unique Class Names {len(class_names)}")
    logger.info(class_names)
    # print(signal_class.iloc[0]['label_names'])

    for i in range(signal_class.shape[0]):
        this_signal = signal_class.iloc[i][1:-5]
        this_signal_name = signal_class.iloc[i]["label_names"]
        if this_signal_name.lower() == "high_signal":
            this_signal_name = "HQS"
        else:
            this_signal_name = "NHQS"

        logger.info(f"The Signal Name {this_signal_name}")
        logger.info(f"The Signal Value {this_signal}")
        this_mel_spectrogram = extract_mel_spectrogram_singular(
            np.array(this_signal).astype(float)
        )

        frequency_spectrum(this_signal, spec=True)

        fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=False, gridspec_kw={'height_ratios': [1, 5]})
        librosa.display.waveshow(np.array(this_signal).astype(float), ax=axes[0], sr=config.SAMPLE_RATE, color='b')
        axes[0].set_title(f"A: {this_signal_name}")
        axes[0].set_ylabel("Pa")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_xticks(np.arange(0, 11))
        img = librosa.display.specshow(
            this_mel_spectrogram, sr=config.SAMPLE_RATE, x_axis="time", y_axis="mel", x_coords=None, y_coords=None,
            hop_length=config.MEL_HOP_LENGTH, n_fft=config.N_FFT,ax=axes[1]
        )
        axes[1].set_title(f"B: {this_signal_name}")
        axes[1].set_ylabel("Mel Frequency (Hz)")
        axes[1].set_xlabel("Time (s)")
        # axes[1].set_yticks(np.linspace(-0.8,50.8,10))
        axes[1].set_xticks(np.arange(0, 11))
        # cbar_ax = fig.add_axes([0.95, 0.115, 0.025, 0.58]) # [left, bottom, width, height] cax=cbar_ax
        plt.colorbar(img, ax=axes[1], format='%+2.0f dB',orientation="vertical", fraction=0.05, pad=0.01) # orientation='horizontal'
        plt.tight_layout()
        # plt.savefig(
        #     f"{config.PARENT_DIRECTORY}/plots/mel_spectrogram_sample/{this_signal_name}_ms_wf_{i}.jpg",
        #     format="jpg",
        #     dpi=300,
        # )
        plt.show()


logger.info("Done")

# plt.show()
# plt.set_title(
#     label=f"Mel Spectrogram {this_signal_name.title()}",
#     fontsize=14,
#     weight="bold",
# )

# plt.savefig(
#     f"{config.PARENT_DIRECTORY}/plots/mel_spectrogram_sample/{this_signal_name}_melspec_{i}.png",
#     format="png",
#     dpi=300,
# )

# this_mfcc = extract_mfcc_singular(np.array(this_signal).astype(float))
# logger.info(this_signal.shape)
# logger.info(this_mel_spectrogram.shape)
# logger.info(this_mfcc.shape)

# plt.figure(figsize=(10, 6))
# librosa.display.specshow(this_mfcc, sr=config.SAMPLE_RATE, x_axis="time")
# plt.colorbar(format="%+2.0f dB")
# plt.title(
#     label=f"MFCC {this_signal_name.title()}",
#     fontsize=20,
#     weight="bold",
# )
# plt.xlabel(xlabel="Time", fontsize=16, weight="bold")
# plt.ylabel(ylabel="Mel Frequency", fontsize=16, weight="bold")
# plt.savefig(
#     f"{config.PARENT_DIRECTORY}/plots/mfcc_sample/{this_signal_name}_mfcc_{i}.png",
#     format="png",
#     dpi=300,
# )


# # plt.text(0, 1.2, 'A', transform=ax[0].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')
# # plt.text(0, 1.2, 'B', transform=ax[1].transAxes, fontsize=20, fontweight='bold', va='top', ha='right')