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
from cnn.cnn import CNNNetworkMeteorological
from utils.helpers import get_logger

logger = get_logger(__file__)

if __name__ == "__main__":
    logger.debug(f"Evaluating Meteorological Sensitive Model")
    # load model
    met_model_path = f"meteorological_model.pth.pth"
    met_model = CNNNetworkMeteorological()
    met_state_dict = torch.load(
        f=f"{config.MODELS_FOLDER}/{met_model_path}",
        weights_only=True,
    )
    met_model.load_state_dict(met_state_dict)

    ############################################################################
    ## METEOROLOGICAL SENSITIVE MODEL: TRAIN
    ############################################################################

    evaluate(
        model=met_model,
        the_transformation=config.TRANSFORMATION,
        x_paths= f"{config.PROCESSED_DATA}/even_odd/x_train.parquet",
        y_paths= f"{config.PROCESSED_DATA}/even_odd/y_train.parquet",
        cm_plot_title= rf"$MC_3$ Confusion Matrix for Train set",
        cm_save_path= "mc3_train_confusion_matrix",
    )

    ############################################################################
    ## METEOROLOGICAL SENSITIVE MODEL: TEST
    ############################################################################

    evaluate(
        model=met_model,
        the_transformation=config.TRANSFORMATION,
        x_paths=f"{config.PROCESSED_DATA}/even_odd/x_test.parquet",
        y_paths=f"{config.PROCESSED_DATA}/even_odd/y_test.parquet",
        cm_plot_title=rf"$MC_3$ Confusion Matrix for Validation set",
        cm_save_path="mc3_validation_confusion_matrix",
    )
    ############################################################################
    ## METEOROLOGICAL SENSITIVE MODEL: TRANSPORTABILITY
    ############################################################################

    evaluate(
        model=met_model,
        the_transformation=config.TRANSFORMATION,
        x_paths=f"{config.PROCESSED_DATA}/even_odd/x_true_test_ds.parquet",
        y_paths=f"{config.PROCESSED_DATA}/even_odd/y_true_test_ds.parquet",
        cm_plot_title=rf"$MC_3$ Confusion Matrix for Transport set",
        cm_save_path="mc3_test_confusion_matrix",
    )

