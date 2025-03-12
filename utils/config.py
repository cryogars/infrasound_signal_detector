"""
Configuration file.
"""

__author__ = "Evi Ofekeze"
__authors__ = ["HP Marshal", "Jefferey B Johnson"]
__contact__ = "eviofekeze@u.boisestate.edu"
__copyright__ = "Copyright 2024, Boise State University, Boise ID"
__group__ = "Cryosphere GeoPhysics and Remote Sensing Laboratory,"
__credits__ = ["Evi Ofekeze", "HP Marshal", "Jefferey B Johnson"]
__email__ = "eviofekeze@u.boisestate.edu"
__maintainer__ = "developer"
__status__ = "Research"

from pathlib import Path
import torch
import torchaudio

# Select Device
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ---------------------------------------------------------------------------------------------

THIS_TRAIN = "cnn"
# THIS_TRAIN = "ffn"

# ---------------------------------------------------------------------------------------------
STATION = "LCC3"  # Either of "LCC2" or "LCC3" Training with LCC2, evaluating with LCC3
SENSOR = 3  # Either of 1, 2, 3 | Training with sensor 3 for LCC2
MODEL_CONFIGURATION = "first_ten"
# MODEL_CONFIGURATION = "even_odd"

# STRATEGY = "under_sampling"
STRATEGY = ""
CLASSES = 4
if CLASSES == 2:
    CLASS_FOLDER = "four_classes"
elif CLASSES == 2:
    CLASS_FOLDER = "two_classes"
else:
    CLASS_FOLDER = "."

# Paths Main and Raw
# PARENT_DIRECTORY = Path(f"/media/eviofekeze/extrastorage/sound/signal_detection")
PARENT_DIRECTORY = Path(f"/bsuhome/eviofekeze/scratch/signal_detection/infrasound_signal_detector")
DATA_PATH_DAYFILES = Path(f"{PARENT_DIRECTORY}/data/dayfiles")
DATA_PATH_STATISTICS = Path(f"{PARENT_DIRECTORY}/data/statistics")
DATA_PATH_EVENTS = Path(f"{PARENT_DIRECTORY}/data/events")
PROCESSED_DATA = Path(f"{PARENT_DIRECTORY}/data/processed")
MODELS_FOLDER = Path(f"{PARENT_DIRECTORY}/models")

WAVEFORM_FILE_FULL_HS = Path(
    f"{PARENT_DIRECTORY}/data/processed/{MODEL_CONFIGURATION}/x_train.parquet"
)
LABEL_FILE_FULL_HS = Path(
    f"{PARENT_DIRECTORY}/data/processed/{MODEL_CONFIGURATION}/y_train.parquet"
)
WAVEFORM_FILE_TEST_FULL_HS = Path(
    f"{PARENT_DIRECTORY}/data/processed/{MODEL_CONFIGURATION}/x_test.parquet"
)
LABEL_FILE_TEST_FULL_HS = Path(
    f"{PARENT_DIRECTORY}/data/processed/{MODEL_CONFIGURATION}/y_test.parquet"
)
#################################################################
# Point to option here
#################################################################
THIS_WAVEFORM_FILE = WAVEFORM_FILE_FULL_HS
THIS_LABEL_FILE = LABEL_FILE_FULL_HS
THIS_WAVEFORM_FILE_TEST = WAVEFORM_FILE_TEST_FULL_HS
THIS_LABEL_FILE_TEST = LABEL_FILE_TEST_FULL_HS

# ---------------------------------------------------------------------------------------------

# Data Processing and Audio Configurations
SAMPLE_RATE = 100  # Infrasound were collected at 100 measurements per seconds
DURATION_IN_SEC = 10  # Data is evaluation at 10 sec slices
SHIFT = 1 * SAMPLE_RATE  # The sliding shift, One sec times measurements per seconds
NUM_SAMPLES = SAMPLE_RATE * DURATION_IN_SEC  # The chunk of audio being considered

# ---------------------------------------------------------------------------------------------
# Transformation Configuration
# ---------------------------------------------------------------------------------------------
N_FFT = 128
MEL_HOP_LENGTH = 32
N_MELS = 32
# ----------------
N_MFCC = 13
DISCRETE_COSINE_TRANSFORM_TYPE = 2
NORM = "ortho"
LOG_MELS = True

# Transformation configuration
TRANSFORMATION_NAME = "mel-spectrogram"  # Either of mel-spectrogram or mfcc
# TRANSFORMATION = "mfcc"
if TRANSFORMATION_NAME == "mfcc":
    this_transformation = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        dct_type=DISCRETE_COSINE_TRANSFORM_TYPE,
        norm=NORM,
        log_mels=LOG_MELS,
    )
elif TRANSFORMATION_NAME == "mel-spectrogram":
    this_transformation = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=MEL_HOP_LENGTH,
        n_mels=N_MELS,
    )
else:
    raise ValueError(
        f"Invalid transformation {TRANSFORMATION_NAME}. "
        f"Specify either of 'MFCC' or 'mel-spectrogram'"
    )

TRANSFORMATION = this_transformation
# Training Params
# ---------------------------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 30

# construct model and assign it to device
if MODEL_CONFIGURATION == "first_ten" or "temporal":
    LEARNING_RATE = 0.0004
elif MODEL_CONFIGURATION == "even_odd" or "meteorological":
    LEARNING_RATE = 0.0004
else:
    LEARNING_RATE = 0.001

MET_LEARNING_RATE = 0.0004
TEMP_LEARNING_RATE = 0.0004
SENSOR_LEARNING_RATE = 0.001

# FFN_LEARNING_RATE = 0.0089

# ---------------------------------------------------------------------------------------------
# LCC Station and Sensors This is useful for adjust for python indexing

# READ! READ!! READ!!!
# Do not Adjust this Chunk Below
# Numbering to fit python indexing
STATION_LCC2 = 1
STATION_LCC3 = 2

INFRASOUND_SENSOR_1 = 0
INFRASOUND_SENSOR_2 = 1
INFRASOUND_SENSOR_3 = 2  # Training with sensor 3 for LCC3

if STATION == "LCC2":
    THIS_STATION = STATION_LCC2
elif STATION == "LCC3":
    THIS_STATION = STATION_LCC3

if SENSOR == 1:
    THIS_SENSOR = INFRASOUND_SENSOR_1
elif SENSOR == 2:
    THIS_SENSOR = INFRASOUND_SENSOR_2
elif SENSOR == 3:
    THIS_SENSOR = INFRASOUND_SENSOR_3
# Do not Adjust this Chunk Above



signal = "high_signal_"

# # Just a Sample for Analysis: Partitioned
# WAVEFORM_FILE_SAMPLE = Path(
#     f"{PARENT_DIRECTORY}/data/processed/sample/{CLASS_FOLDER}/{signal}X_train_{STRATEGY}.parquet"
# )
# LABEL_FILE_SAMPLE = Path(
#     f"{PARENT_DIRECTORY}/data/processed/sample/{CLASS_FOLDER}/{signal}y_train_{STRATEGY}.parquet"
# )
# WAVEFORM_FILE_TEST_SAMPLE = Path(
#     f"{PARENT_DIRECTORY}/data/processed/sample/{CLASS_FOLDER}/{signal}X_test_{STRATEGY}.parquet"
# )
# LABEL_FILE_TEST_SAMPLE = Path(
#     f"{PARENT_DIRECTORY}/data/processed/sample/{CLASS_FOLDER}/{signal}y_test_{STRATEGY}.parquet"
# )
#
#
# # Full Data: Partitioned to minority class
# WAVEFORM_FILE_FULL = Path(
#     f"{PARENT_DIRECTORY}/data/processed/{CLASS_FOLDER}/full_X_train_{STRATEGY}.parquet"
# )
# LABEL_FILE_FULL = Path(
#     f"{PARENT_DIRECTORY}/data/processed/{CLASS_FOLDER}/full_y_train_{STRATEGY}.parquet"
# )
# WAVEFORM_FILE_TEST_FULL = Path(
#     f"{PARENT_DIRECTORY}/data/processed/{CLASS_FOLDER}/full_X_test_{STRATEGY}.parquet"
# )
# LABEL_FILE_TEST_FULL = Path(
#     f"{PARENT_DIRECTORY}/data/processed/{CLASS_FOLDER}/full_y_test_{STRATEGY}.parquet"
# )
#
#
# # Full Data: Partitioned to minority class
# WAVEFORM_FILE_FULL_HS = Path(
#     f"{PARENT_DIRECTORY}/data/processed/two_high_signal_half/high_signal_X_train_.parquet"
# )
# LABEL_FILE_FULL_HS = Path(
#     f"{PARENT_DIRECTORY}/data/processed/two_high_signal_half/high_signal_y_train_.parquet"
# )
# WAVEFORM_FILE_TEST_FULL_HS = Path(
#     f"{PARENT_DIRECTORY}/data/processed/two_high_signal_half/high_signal_X_test_.parquet"
# )
# LABEL_FILE_TEST_FULL_HS = Path(
#     f"{PARENT_DIRECTORY}/data/processed/two_high_signal_half/high_signal_y_test_.parquet"
# )
#