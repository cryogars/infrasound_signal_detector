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
sys.path.append("../")

import torch
import torchaudio

from evaluaton.evaluation import evaluate
from utils import config as config
from cnn.cnn import CNNNetwork, CNNNetworkTemporal, CNNNetworkMeteorological

from utils.helpers import get_logger

logger = get_logger(__file__)

if __name__ == "__main__":

    # load model
    logger.debug(f"Evaluating Sensor Agnostic Model L3")
    logger.debug(f"Loading Model")
    the_model_path = f"sensor_model_l3_s3.pth"
    the_model = CNNNetwork()
    the_state_dict = torch.load(
        f=f"{config.MODELS_FOLDER}/{the_model_path}",
        weights_only=True,
    )
    the_model.load_state_dict(the_state_dict)

    ############################################################################
    ## SENSOR AGNOSTIC MODEL: TRAIN
    ############################################################################
    evaluate(
        model=the_model,
        the_transformation=config.TRANSFORMATION,
        x_paths= f"{config.PROCESSED_DATA}/two_high_signal_l3s3/high_signal_X_train_.parquet",
        y_paths= f"{config.PROCESSED_DATA}/two_high_signal_l3s3/high_signal_y_train_.parquet",
        cm_plot_title= rf"$MC_1$ Confusion Matrix for Train set L3S3",
        cm_save_path= "mc1_train_confusion_matrix_l3s3",
    )

    ############################################################################
    ## SENSOR AGNOSTIC MODEL: TEST
    ############################################################################
    evaluate(
        model=the_model,
        the_transformation=config.TRANSFORMATION,
        x_paths=f"{config.PROCESSED_DATA}/two_high_signal_l3s3/high_signal_X_test_.parquet",
        y_paths=f"{config.PROCESSED_DATA}/two_high_signal_l3s3/high_signal_y_test_.parquet",
        cm_plot_title=rf"$MC_1$ Confusion Matrix for Validation set L3S3",
        cm_save_path="mc1_validation_confusion_matrix_l3s3",
    )

    ############################################################################
    ## SENSOR AGNOSTIC MODEL: TRANSPORTABILITY
    ############################################################################

    stations = ["LCC2", "LCC3"]
    sensors = [1, 2, 3]
    for station in stations:
        for sensor in sensors:
            if not (station == "LCC3" and sensor == 3):
                logger.debug(f"Evaluating Station {station}, Sensor: {sensor}")
                happy_path = f"{config.PROCESSED_DATA}/two_high_signal/"
                X_path = f"{happy_path}/{station}/X_sensor{sensor}_full.parquet"
                y_path = f"{happy_path}/{station}/y_sensor{sensor}_full.parquet"

                logger.debug(
                    f"Evaluation for Model {the_model_path}, Station: {station}, Sensor: {sensor}"
                )

                evaluate(
                    model=the_model,
                    the_transformation=config.TRANSFORMATION,
                    x_paths=X_path,
                    y_paths=y_path,
                    cm_plot_title=rf"$MC_1$ Confusion Matrix: Station {station} Sensor: {sensor} L3S3",
                    cm_save_path=f"mc1_cm_station{station}_sensor{sensor}_l3s3",
                )
