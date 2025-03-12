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
from cnn.cnn import CNNNetworkTemporal
from utils.helpers import get_logger

logger = get_logger(__file__)

if __name__ == "__main__":
    # load model

    logger.debug(f"Evaluating Meteorological Temporal Sensitive Model")
    temp_model_path = f"temporal_model.pth.pth"
    temp_model = CNNNetworkTemporal()
    temp_state_dict = torch.load(
        f=f"{config.MODELS_FOLDER}/{temp_model_path}",
        weights_only=True,
    )
    temp_model.load_state_dict(temp_state_dict)

    ############################################################################
    ## TEMPORAL SENSITIVE MODEL: TRAIN
    ############################################################################

    evaluate(
        model=temp_model,
        the_transformation=config.TRANSFORMATION,
        x_paths= f"{config.PROCESSED_DATA}/first_ten/x_train.parquet",
        y_paths= f"{config.PROCESSED_DATA}/first_ten/y_train.parquet",
        cm_plot_title= rf"$MC_2$ Confusion Matrix for train set",
        cm_save_path= "mc2_train_confusion_matrix",
    )

    ############################################################################
    ## TEMPORAL SENSITIVE MODEL: TEST
    ############################################################################

    evaluate(
        model=temp_model,
        the_transformation=config.TRANSFORMATION,
        x_paths=f"{config.PROCESSED_DATA}/first_ten/x_test.parquet",
        y_paths=f"{config.PROCESSED_DATA}/first_ten/y_test.parquet",
        cm_plot_title=rf"$MC_2$ Confusion Matrix for Validation set",
        cm_save_path="mc2_validation_confusion_matrix",
    )

    ############################################################################
    ## TEMPORAL SENSITIVE MODEL: TRANSPORTABILITY
    ############################################################################

    evaluate(
        model=temp_model,
        the_transformation=config.TRANSFORMATION,
        x_paths=f"{config.PROCESSED_DATA}/first_ten/x_true_test_ds.parquet",
        y_paths=f"{config.PROCESSED_DATA}/first_ten/y_true_test_ds.parquet",
        cm_plot_title=rf"$MC_2$ Confusion Matrix for Transport set",
        cm_save_path="mc2_test_confusion_matrix",
    )

