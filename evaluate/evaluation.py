"""
Evaluation
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

import torch
import torchaudio

import sys
sys.path.append("../")


from utils import config as config
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from cnn.cnn import CNNNetwork, CNNNetworkTemporal, CNNNetworkMeteorological
from utils.infrasounddataset import InfrasoundDataset
from utils.helpers import get_logger

logger = get_logger(__file__)


if config.CLASSES == 4:
    class_mapping = [
        "ambiguous",
        "signal",
        "high_signal",
        "noise",
    ]
else:
    class_mapping = [
        "noise",
        "high_signal",
    ]


def predict(model, _input, _target, _class_mapping):
    model.eval()

    with torch.no_grad():
        predictions = model(_input)
        predicted_index = predictions[0].argmax(0)
        _predicted = _class_mapping[predicted_index]
        _expected = _class_mapping[_target]

    return _predicted, _expected


def target_array(model, data, _class_mapping=None):
    if _class_mapping is None:
        _class_mapping = class_mapping

    predicted_array, expected_array = [], []

    for i in range(len(data)):
        _input, _target = data[i][0], data[i][1]
        _input.unsqueeze_(0)

        # make an inference
        predicted, expected = predict(
            model=model, _input=_input, _target=_target, _class_mapping=class_mapping
        )
        predicted_array.append(predicted)
        expected_array.append(expected)
    return predicted_array, expected_array


def evaluate(
    model,
    the_transformation,
    x_paths: str = None,
    y_paths: str = None,
    cm_plot_title: str = None,
    cm_save_path: str =None,
) -> None:

    isd = InfrasoundDataset(
        labels_file=y_paths,
        waveform_file=x_paths,
        transformation=the_transformation,
        target_sample_rate=config.SAMPLE_RATE,
        num_samples=config.NUM_SAMPLES,
        device="cpu",
    )

    predicted_array, expected_array = target_array(
        model=model, data=isd, _class_mapping=class_mapping
    )
    this_report = classification_report(expected_array, predicted_array)
    logger.info(f"Classification Report {cm_plot_title.title()} DataSet\n{this_report}")
    this_cm = confusion_matrix(expected_array, predicted_array)

    cm_percent = this_cm.astype(np.float64) / this_cm.sum(axis=1, keepdims=True) * 100

    # ConfusionMatrixDisplay(this_cm).plot(values_format="d")
    class_labels = ["HQS", "NHQS"]
    disp = ConfusionMatrixDisplay(confusion_matrix=this_cm, display_labels=class_labels)

    fig, ax = plt.subplots(figsize=(5, 5))

    # for text in ax.texts:
    #     text.set_visible(False)

    disp.plot(ax=ax, colorbar=False, values_format="d")
    for i in range(this_cm.shape[0]):
        for j in range(this_cm.shape[1]):
            count = this_cm[i, j]
            percent = cm_percent[i, j]
            ax.text(j, i, f"\n \n({percent:.1f}%)",
                    ha="center", va="center",fontsize=10)

    plt.xlabel('Predicted Labels', fontsize='12', fontweight='bold')
    plt.ylabel('True Labels', fontsize='12', fontweight='bold')
    plt.title(cm_plot_title, fontsize='12', fontweight='bold')
    save_path = f"{config.PARENT_DIRECTORY}/plots/cm/{cm_save_path}.png"
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )




