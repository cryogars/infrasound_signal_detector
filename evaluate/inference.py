"""
Inference
"""

__author__ = "Evi Ofekeze"
__authors__ = ["HP Marshal", "Jefferey B Johnson"]
__contact__ = "eviofekeze@u.boisestate.edu"
__copyright__ = "Copyright 2024, Boise State University, Boise ID"
__group__ = "Cryosphere GeoPhysics and Remote Sensing Laboratory"
__credits__ = ["Evi Ofekeze", "HP Marshall", "Jeffrey B. Johnson"]
__email__ = "eviofekeze@u.boisestate.edu"
__maintainer__ = "developer"
__status__ = "Research"


import torch
import torchaudio
from cnn.cnn import CNNNetwork
from utils.infrasounddataset import InfrasoundDataset
import utils.config as config
from utils.helpers import get_logger

logger = get_logger(__file__)


if config.CLASSES == 4:
    class_mapping = [
        "ambiguous",
        "signal",
        "high-signal",
        "noise",
    ]
else:
    class_mapping = [
        "noise",
        "high-signal",
    ]


def predict(model, _input, _target, _class_mapping):
    model.eval()

    with torch.no_grad():
        predictions = model(_input)
        predicted_index = predictions[0].argmax(0)
        _predicted = _class_mapping[predicted_index]
        _expected = _class_mapping[_target]

    return _predicted, _expected


if __name__ == "__main__":
    # load model
    if config.THIS_TRAIN == "cnn":
        the_model = CNNNetwork()
    elif config.THIS_TRAIN == "ffn":
        the_model = FFNNetwork()
    else:
        raise Exception(f"Specify model in config file")
    state_dict = torch.load(
        f"{config.MODELS_FOLDER}/{config.THIS_TRAIN}-{config.TRANSFORMATION}_{config.STRATEGY}.pth",
        weights_only=True,
    )
    logger.debug(
        f"{config.MODELS_FOLDER}/{config.THIS_TRAIN}-{config.TRANSFORMATION}_{config.STRATEGY}.pth"
    )
    the_model.load_state_dict(state_dict)

    if config.TRANSFORMATION == "mfcc":
        this_transformation = torchaudio.transforms.MFCC(
            sample_rate=config.SAMPLE_RATE,
            n_mfcc=config.N_MFCC,
            dct_type=config.DISCRETE_COSINE_TRANSFORM_TYPE,
            norm=config.NORM,
            log_mels=config.LOG_MELS,
        )
    elif config.TRANSFORMATION == "mel-spectrogram":
        this_transformation = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.MEL_HOP_LENGTH,
            n_mels=config.N_MELS,
        )
    else:
        raise ValueError(
            f"Invalid transformation {config.TRANSFORMATION}. "
            f"Specify either of 'MFCC' or 'mel-spectrogram'"
        )

    isd = InfrasoundDataset(
        labels_file=config.THIS_LABEL_FILE,
        waveform_file=config.THIS_WAVEFORM_FILE,
        transformation=this_transformation,
        target_sample_rate=config.SAMPLE_RATE,
        num_samples=config.NUM_SAMPLES,
        device="cpu",
    )

    _input, _target = isd[1][0], isd[1][1]  # [instances][corresponding target]
    _input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(
        model=the_model, _input=_input, _target=_target, _class_mapping=class_mapping
    )
    logger.info(f"Predicted: '{predicted}', expected: '{expected}'")
