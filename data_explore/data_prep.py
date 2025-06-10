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
sys.path.append("../")

import gc
from pathlib import Path
import scipy.io as sio
import pandas as pd

# User Defined libraries
import utils.config as config
from utils.helpers import (
    get_logger,
    create_slides_np,
    create_dataframe,
    partition_data,
    create_full_dataset,
    partition_high_signal,
)

logger = get_logger(__file__)


def log_and_save(
    x_train,
    y_train,
    x_test,
    y_test,
    root_path: str,
    sample_style: str,
    this_case: str,
    full_not_full: str,
) -> None:
    logger.info(f"Shape of X_train({this_case}): {x_train.shape}")
    logger.info(f"Shape of y_train({this_case}): {y_train.shape}")
    logger.info(f"Shape of X_test({this_case}): {x_test.shape}")
    logger.info(f"Shape of y_test({this_case}): {y_test.shape}")

    logger.info(f"Saving partitioned {this_case}  in Parquet")
    partitioned_path = Path(
        f"{root_path}/{full_not_full}X_train_{sample_style}.parquet"
    )
    if partitioned_path.is_file():
        logger.info(f"File already exist. Moving on ...")
    else:
        x_train.to_parquet(
            f"{root_path}/{full_not_full}X_train_{sample_style}.parquet",
            compression="gzip",
        )
        x_test.to_parquet(
            f"{root_path}/{full_not_full}X_test_{sample_style}.parquet",
            compression="gzip",
        )
        y_train.to_parquet(
            f"{root_path}/{full_not_full}y_train_{sample_style}.parquet",
            compression="gzip",
        )
        y_test.to_parquet(
            f"{root_path}/{full_not_full}y_test_{sample_style}.parquet",
            compression="gzip",
        )
        logger.info(f"Saved partitioned {this_case} data as Parquet")
        return


if __name__ == "__main__":


    logger.info(f"BEGIN SAMPLE DATA PROCESSING")
    #############################################################
    # Loading WaveForm Data
    #############################################################

    logger.info(f"Begin load data: Raw Waveform")
    waveform_raw = sio.loadmat(f"{config.DATA_PATH_DAYFILES}/081data2023Mar22.mat")
    logger.info(f"Waveform Keys: {waveform_raw.keys()}")

    waveform = waveform_raw["acfilts"][:, config.THIS_SENSOR, config.THIS_STATION]
    logger.info(f"Original Waveform Shape: {waveform.shape}")

    waveform2d = create_slides_np(
        some_array=waveform_raw,
        station_id=config.THIS_STATION,
        sensor_id=config.THIS_SENSOR,
        strides=config.SHIFT,
        window_length=config.NUM_SAMPLES,
    )
    logger.info(f"New Waveform Shape: {waveform2d.shape}")
    logger.info(f"End load data: Raw Waveform")
    gc.collect()

    #############################################################
    # Loading Statistics
    #############################################################

    logger.info(f"Begin Load Data, Statistics")

    waveform_stats = sio.loadmat(f"{config.DATA_PATH_STATISTICS}/hist2D081.mat")

    logger.info(f"Waveform Statistics Keys: {waveform_stats.keys()}")
    # logger.info(
    #     f"Waveform Consistency Shape: {waveform_stats["const"][:, config.THIS_STATION].shape}"
    # )
    # logger.info(
    #     f"Waveform Correlation Shape: {waveform_stats["xcth"][:, config.THIS_STATION].shape}"
    # )
    # logger.info(f"Waveform Samples Shape: {waveform_stats["samples"][0].shape}")
    logger.info(f"End Load Data, Statistics")
    # gc.collect()

    #############################################################
    # Creating DataFrame
    #############################################################

    waveform_df = create_dataframe(
        waveform_arr=waveform2d,
        statistics_dict=waveform_stats,
        station=config.THIS_STATION,
    )

    logger.info(f"Data Frame Head\n{waveform_df.head()}")
    logger.info(
        f"Descriptive Statistics for Consistently and Correlation\n"
        f"{waveform_df[['const', 'xcth', 'label']].describe()}"
    )
    print(waveform_df.label.unique())

    #############################################################
    # Saving DataFrame
    #############################################################
    #
    logger.info(f"Saving DataFrame in Parquet")
    this_path = Path(
        f"{config.PARENT_DIRECTORY}/data/sample/waveform_stats_df_histl3.parquet"
    )
    if this_path.is_file():
        logger.info(f"File already exist. Moving on ...")
    else:
        waveform_df.to_parquet(
            path=Path(
                f"{config.PARENT_DIRECTORY}/data/sample//waveform_stats_df_histl3.parquet"
            ),
            compression="gzip",
        )
        logger.info(f"Saved DataFrame Parquet")
    gc.collect()

    #############################################################
    # Partitioning Data
    #############################################################

    # labels_to_drop = []  # 'ambiguous'
    # waveform_df_filtered = (
    #     waveform_df  # .loc[~waveform_df['label_names'].isin(labels_to_drop)]
    # )
    # result = partition_data(df=waveform_df_filtered, sampling_strategy="")
    # X_train, X_test, y_train, y_test = (
    #     result.x_train,
    #     result.x_test,
    #     result.y_train,
    #     result.y_test,
    # )
    # sample_path = f"{config.PROCESSED_DATA}/sample/{config.CLASS_FOLDER}"
    # log_and_save(
    #     x_train=X_train,
    #     y_train=y_train,
    #     x_test=X_test,
    #     y_test=y_test,
    #     sample_style="",
    #     root_path=sample_path,
    #     this_case="SAMPLE DATA CASE",
    #     full_not_full="",
    # )
    #
    # gc.collect()
    #############################################################
    # Partitioning undersample Data
    #############################################################
    #
    # labels_to_drop = []  # 'ambiguous'
    # waveform_df_filtered = (
    #     waveform_df  # .loc[~waveform_df['label'].isin(labels_to_drop)]
    # )
    # X_train, X_test, y_train, y_test = partition_data(
    #     df=waveform_df_filtered, sampling_strategy="under_sampling"
    # )
    #
    # sample_path = f"{config.PROCESSED_DATA}/sample/{config.CLASS_FOLDER}"
    # log_and_save(
    #     x_train=X_train,
    #     y_train=y_train,
    #     x_test=X_test,
    #     y_test=y_test,
    #     sample_style="under_sampling",
    #     root_path=sample_path,
    #     this_case="SAMPLE DATA CASE",
    #     full_not_full="",
    # )
    #
    # logger.info(f"END SAMPLE DATA PROCESSING")
    # gc.collect()

    #############################################################
    # Create full DataFrame
    #############################################################
    # logger.info(f"BEGIN FULL DATA PROCESSING")
    # logger.info(f"Begin create full data frame for {config.THIS_STATION} and sensor {config.THIS_SENSOR}")
    #
    # this_path = Path(
    #     f"{config.PROCESSED_DATA}/two_high_signal_l3s3/full_waveform_stats_l3s3_df.parquet"
    # )
    # if this_path.is_file():
    #     logger.info(f"File {this_path} already exist. Moving on ...")
    # else:
    #     full_data = create_full_dataset(
    #         waveform_dir=config.DATA_PATH_DAYFILES,
    #         statistics_dir=config.DATA_PATH_STATISTICS,
    #         station=config.THIS_STATION,
    #         sensor=config.THIS_SENSOR,
    #     )
    #     logger.info(f"Saving DataFrame in Parquet")
    #     full_data.to_parquet(path=this_path, compression="gzip")
    #     logger.info(f"Saved DataFrame Parquet")
    #
    # logger.info(f"End create full data frame")
    #############################################################
    # Partition full DataFrame
    #############################################################
    # gc.collect()
    # logger.info(f"Begin partitioning full data frame")
    # X_train, X_test, y_train, y_test = partition_data(
    #     df=full_data, sampling_strategy=""
    # )
    #
    # sample_path = f"{config.PROCESSED_DATA}/{config.CLASS_FOLDER}"
    # log_and_save(
    #     x_train=X_train,
    #     y_train=y_train,
    #     x_test=X_test,
    #     y_test=y_test,
    #     sample_style="",
    #     root_path=sample_path,
    #     this_case="FULL DATA CASE",
    #     full_not_full="full_",
    # )
    #
    # logger.info(f"End partitioning full data frame")
    # gc.collect()
    #############################################################
    # Undersample full DataFrame
    #############################################################
    # logger.info(f"Begin partitioning under sampled data frame")
    # import pandas as pd
    #
    # full_data = pd.read_parquet(
    #     f"{config.PROCESSED_DATA}/{config.CLASS_FOLDER}/full_waveform_stats_df.parquet"
    # )
    # print(full_data["label"].value_counts())
    #
    # gc.collect()
    # X_train, X_test, y_train, y_test = partition_data(
    #     df=full_data, sampling_strategy="under_sampling"
    # )
    #
    # del full_data
    #
    # sample_path = f"{config.PROCESSED_DATA}/{config.CLASS_FOLDER}"
    # log_and_save(
    #     x_train=X_train,
    #     y_train=y_train,
    #     x_test=X_test,
    #     y_test=y_test,
    #     sample_style="under_sampling",
    #     root_path=sample_path,
    #     this_case="FULL DATA CASE",
    #     full_not_full="full_",
    # )
    #
    # logger.info(f"Completed partitioning of under sampled data frame")

    ############################################################
    # Create Full Data
    #############################################################

    # logger.info(f"BEGIN FULL DATA PROCESSING")
    # logger.info(f"Begin create full data frame")
    #
    # full_data = create_full_dataset(
    #     waveform_dir=config.DATA_PATH_DAYFILES,
    #     statistics_dir=config.DATA_PATH_STATISTICS,
    #     station=config.THIS_STATION,
    #     sensor=config.THIS_SENSOR,
    # )
    #
    # logger.info(f"processing station: {config.STATION} sensor: {config.SENSOR}")
    # logger.info(f"Saving DataFrame in Parquet")
    # this_path = Path(
    #     f"{config.PROCESSED_DATA}/two_high_signal/full_waveform_stats_df_{config.STATION}_test_sensor_{config.SENSOR}.parquet"
    # )
    # if this_path.is_file():
    #     logger.info(f"File {this_path} already exist. Moving on ...")
    # else:
    #     full_data.to_parquet(path=this_path, compression="gzip")
    #     logger.info(f"Saved DataFrame Parquet")
    #
    # logger.info(f"End create full data frame")

    ############################################################
    # Partitioning High Signal
    #############################################################

    # full_data = pd.read_parquet(
    #     f"{config.PROCESSED_DATA}/two_high_signal/LCC3/full_waveform_stats_df_LCC3_test_sensor_3.parquet"
    # )
    #
    # result = partition_high_signal(df=full_data)
    # X_train, X_test, y_train, y_test = (
    #     result.x_train,
    #     result.x_test,
    #     result.y_train,
    #     result.y_test,
    # )
    #
    # sample_path = f"{config.PROCESSED_DATA}/two_high_signal_l3s3"
    # log_and_save(
    #     x_train=X_train,
    #     y_train=y_train,
    #     x_test=X_test,
    #     y_test=y_test,
    #     sample_style="",
    #     root_path=sample_path,
    #     this_case="HIGH SIGNAL CASE",
    #     full_not_full="high_signal_",
    # )
    #
    # fdf = result.df
    # save_path = f"{config.PROCESSED_DATA}/two_high_signal_l3s3/fdf"
    # fdf.to_parquet(path=save_path, compression="gzip")

    ############################################################
    # End of Script
    #############################################################
