"""
helper functions
"""

__author__ = "Evi Ofekeze"
__authors__ = ["HP Marshal", "Jefferey B Johnson"]
__contact__ = "eviofekeze@u.boisestate.edu"
__copyright__ = "Copyright 2024, Boise State University, Boise ID"
__credits__ = ["Evi Ofekeze", "HP Marshal", "Jefferey B Johnson"]
__email__ = "eviofekeze@u.boisestate.edu"
__maintainer__ = "developer"
__status__ = "Research"

import sys
sys.path.append('../')

import gc
import pathlib
from collections import namedtuple
from typing import Union, NamedTuple

# Base Libraries : Plotting | Data Wrangling | Scientific Data Manipulation
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# System utility libraries
import logging
import os
from pathlib import Path


import utils.config as config


# Setting up logger
def get_logger(file_name):
    """
    :param file_name: Name file
    :return: Logger object
    """
    logger_name = os.path.basename(file_name).split(".")[0]
    logger = logging.getLogger(logger_name.upper())
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(name)s - %(asctime)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(stream_handler)

    return logger


logger_h = get_logger(__file__)


def create_slides_np(
    some_array: dict,
    station_id: int = config.THIS_STATION,
    sensor_id: int = config.THIS_SENSOR,
    strides: int = config.SHIFT,
    window_length: int = config.NUM_SAMPLES,
) -> np.array:
    """
    Function to create chunks in audio data with one sec time step and 10 seconds chunks
    :param sensor_id:
    :param station_id:
    :param some_array: the flat infrasound array
    :param strides: the time step
    :param window_length: the audio chunk length
    :return: A 2D array of data
    """
    # n_rows = (len(some_array) - 1000) // (100 + 1)
    return np.lib.stride_tricks.sliding_window_view(
        some_array["acfilts"][:, sensor_id, station_id], window_shape=window_length
    )[::strides]


def signal_labeller(df: pd.DataFrame) -> int:
    """
    Function to apply labels to dataframe
    :param df: the dataframe
    :return: labels for data frame
    """
    consistency, correlation = df["const"], df["xcth"]
    if config.CLASSES == 4:
        if consistency <= 5 and correlation >= 0.80:
            return 2  # high_signal
        elif consistency <= 5 and (0.80 > correlation > 0.1):
            return 1  # signal
        elif consistency >= 25 and (0.1 < correlation < 0.6):
            return 3  # noise
        else:
            return 0  # ambiguous
    elif config.CLASSES == 2:
        if consistency <= 5:
            return 0  # signal
        else:
            return 1  # noise


def create_dataframe(
    waveform_arr: np.ndarray, statistics_dict: dict, station=config.THIS_STATION
) -> pd.DataFrame:
    """
    :param waveform_arr: raw waveform data
    :param statistics_dict: statistics Data
    :param station: infrasound Station ID for Stats retrival
    :return:
    """
    column_names = [
        f"M{(i % config.SAMPLE_RATE) + 1}S{(i // config.SAMPLE_RATE) + 1}"
        for i in range(waveform_arr.shape[1])
    ]

    waveform_df = pd.DataFrame(waveform_arr, columns=column_names)
    waveform_df["const"] = statistics_dict["const"][:, station]
    waveform_df["xcth"] = statistics_dict["xcth"][:, station]
    waveform_df["samples"] = statistics_dict["samples"][0]
    waveform_df["label"] = waveform_df.apply(signal_labeller, axis=1)
    if config.CLASSES == 4:
        label_mapping = {0: "ambiguous", 1: "signal", 2: "high_signal", 3: "noise"}
    else:
        label_mapping = {0: "high_signal", 1: "noise"}

    waveform_df["label_names"] = waveform_df["label"].map(label_mapping)
    return waveform_df


def create_full_dataset(
    waveform_dir: pathlib.PosixPath,
    statistics_dir: pathlib.PosixPath,
    station: int,
    sensor: int,
    logger: logging.Logger = logger_h,
) -> pd.DataFrame:
    """
    :param logger: The logging object
    :param waveform_dir: Directory for raw waveform data
    :param statistics_dir: directory for the stats variable, consistency and correlation
    :param station: The station of reference
    :param sensor:  The sensor of reference
    :return: the full dataframe that combined the full study period
    """

    all_waveform_filename = [f.name for f in waveform_dir.iterdir() if f.is_file()]
    combined_df = None
    julian_days = [julian_day[:3] for julian_day in all_waveform_filename]
    for i in range(len(julian_days)):
        julian_day = julian_days[i]

        waveform_file_path = [
            string for string in all_waveform_filename if string.startswith(julian_day)
        ][0]

        logger.info(f"Processing Julian day {julian_day}")
        waveform_file = Path(f"{waveform_dir}/{waveform_file_path}")
        labels_file = Path(f"{statistics_dir}/hist2D{julian_day}.mat")

        waveform_raw = sio.loadmat(str(waveform_file))
        waveform_stats = sio.loadmat(str(labels_file))

        waveform2d = create_slides_np(
            some_array=waveform_raw,
            station_id=station,
            sensor_id=sensor,
            strides=config.SHIFT,
            window_length=config.NUM_SAMPLES,
        )

        this_df = create_dataframe(
            waveform_arr=waveform2d,
            statistics_dict=waveform_stats,
            station=config.THIS_STATION,
        )

        combined_df = pd.concat([combined_df, this_df], axis=0)
        logger.info(f"Completed Julian day {julian_day} processing")
        logger.info(f"Combined Dataframe Rows Current: {combined_df.shape[0]} ")
        logger.info(f"Combined Dataframe Rows Expected: {86391 * (i + 1)}")
    logger.info(f"Completed Data Creation")
    combined_df = combined_df.reset_index(drop=True)
    return combined_df


def partition_data(df: pd.DataFrame, sampling_strategy: str) -> namedtuple:
    """
    helper function to oversample or under sample data
    :param df: Pandas Dataframe of Labeled Inputs
    :return: named tuple of the partition set
    :param sampling_strategy: {under_sampling, over_sampling}, default: 'None'
                                over_sampling: creates synthetic minority class
                                under_sampling: reduces majority class
    :return: named tuple of the partition set
    """
    Result = namedtuple(
        typename="Result", field_names=["x_train", "x_test", "y_train", "y_test"]
    )
    if sampling_strategy:
        class_size = (
            df["label_names"].value_counts().min()
            if sampling_strategy == "under_sampling"
            else df["label_names"].value_counts().max()
        )
        sampled_list = []
        for class_label in df["label_names"].unique():
            class_subset = df[df["label_names"] == class_label]
            sampled_label = resample(
                class_subset, replace=False, n_samples=class_size, random_state=42
            )
            sampled_list.append(sampled_label)
        balanced_data = pd.concat(sampled_list)
        del sampled_list
        gc.collect()
    else:
        balanced_data = df.copy()
        del df
        gc.collect()

    x = balanced_data.drop(columns=["const", "samples", "xcth", "label", "label_names"])
    y = balanced_data[["label"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    del balanced_data
    gc.collect()
    return Result(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


def partition_high_signal(df: pd.DataFrame) -> namedtuple:
    """
    helper function to oversample or under sample data
    :param df: Pandas Dataframe of Labeled Inputs
    :return: named tuple of the partition set
    """
    Result = namedtuple(
        typename="Result", field_names=["df", "x_train", "x_test", "y_train", "y_test"]
    )

    class_size = df["label_names"].value_counts().get("high_signal")
    # class_size = (df["label_names"] == "high_signal").sum()
    logger_h.info(f"high signal class size : {class_size}")
    print(f"high signal class size : {class_size}")
    # class_size = int(class_size / 0.5)
    high_signal_class = df[df["label_names"] == "high_signal"]
    sampled_list = [high_signal_class]

    class_labels = list(df["label_names"].unique())
    class_labels.remove("high_signal")

    for class_label in class_labels:
        class_subset = df[df["label_names"] == class_label]
        sampled_label = resample(
            class_subset, replace=False, n_samples=class_size, random_state=42
        )
        sampled_list.append(sampled_label)
    balanced_data = pd.concat(sampled_list)
    del sampled_list
    gc.collect()

    balanced_data["label"] = balanced_data["label_names"].apply(
        lambda label: 1 if label == "high_signal" else 0
    )

    balanced_data["label_names"] = balanced_data["label_names"].apply(
        lambda label: label if label == "high_signal" else "noise"
    )

    x = balanced_data.drop(columns=["const", "samples", "xcth", "label", "label_names"])
    y = balanced_data[["label"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    gc.collect()
    return Result(
        df=balanced_data, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test
    )




def make_df(
    this_partition_list,
    logger: logging.Logger = logger_h,
):
    combined_df = None

    logger.info(f"Retrieve julian day codes")
    julian_days = [train_julian_day[:3] for train_julian_day in this_partition_list]

    logger.debug(f"processing{julian_days}")

    logger.info(f"iterate through all days in list to make data frame")

    for day in julian_days:
        waveform_file_path = [
            string for string in this_partition_list if string.startswith(day)
        ][0]

        logger.debug(f"processing path: {waveform_file_path}")

        waveform_file = Path(f"../data/dayfiles/{waveform_file_path}")
        labels_file = Path(f"../data/statistics/hist2D{day}.mat")

        waveform_raw = sio.loadmat(str(waveform_file))
        waveform_stats = sio.loadmat(str(labels_file))

        stations = [0, 1]
        sensors = [0, 1, 2]

        for station in stations:
            for sensor in sensors:
                logger.info(
                    f"Procession day: {day}, station: {station} and sensor: {sensor}"
                )
                waveform2d = create_slides_np(
                    some_array=waveform_raw,
                    station_id=station,
                    sensor_id=sensor,
                    strides=config.SHIFT,
                    window_length=config.NUM_SAMPLES,
                )

                this_df = create_dataframe(
                    waveform_arr=waveform2d,
                    statistics_dict=waveform_stats,
                    station=station,
                )
                logger_h.info(f"Combining data frame")
                combined_df = pd.concat([combined_df, this_df], axis=0)
    logger.info(f"Resetting Index")
    combined_df = combined_df.reset_index(drop=True)
    return combined_df


def create_temporal_coverage_data(
    waveform_dir: pathlib.Path,
    this_case: "str" = "first_ten",
    logger: logging.Logger = logger_h,
):
    """
    :param this_case:
    :param waveform_dir:
    :param logger:
    :return:
    """

    logger.info(f"Begin create temporal data")
    DF_Result = namedtuple(typename="DF_Result", field_names=["train", "test"])

    all_waveform_filename = [f.name for f in waveform_dir.iterdir() if f.is_file()]
    all_waveform_filename.sort()

    logger.info(f"Processing case {this_case}")

    if this_case == "even_odd":
        train_list = [item for item in all_waveform_filename if int(item[:3]) % 2 == 1]
        test_list = [item for item in all_waveform_filename if int(item[:3]) % 2 == 0]
    else:
        train_list = all_waveform_filename[:13]
        test_list = all_waveform_filename[13:]

    train_df = make_df(this_partition_list=train_list)
    test_df = make_df(this_partition_list=test_list)

    return DF_Result(train=train_df, test=test_df)


if __name__ == "__main__":
    ...


    # logger_h.info(f"Begin Even Odd")
    # df_path = Path("../data/processed/even_odd/even_odd_train.parquet")
    # df = pd.read_parquet(df_path)
    # data = partition_high_signal(df)
    #
    # logger_h.info(f"Begin x_train")
    # data.x_train.to_parquet(
    #     f"../data/processed/even_odd/x_train.parquet",
    #     compression="gzip",
    # )
    #
    # logger_h.info(f"Begin y_train")
    # data.y_train.to_parquet(
    #     f"../data/processed/even_odd/y_train.parquet",
    #     compression="gzip",
    # )
    #
    # logger_h.info(f"Begin x_test")
    # data.x_test.to_parquet(
    #     f"../data/processed/even_odd/x_test.parquet",
    #     compression="gzip",
    # )
    #
    # logger_h.info(f"Begin y_test")
    # data.y_test.to_parquet(
    #     f"../data/processed/even_odd/y_test.parquet",
    #     compression="gzip",
    # )
    #
    # logger_h.info(f"Begin DF")
    # data.df.to_parquet(
    #     f"../data/processed/even_odd/balanced_data_eo.parquet",
    #     compression="gzip",
    # )
    #
    # # -------------------------------------------------------------------------
    # df_path_f10 = Path("../data/processed/first_ten/first_ten_train.parquet")
    # df_f10 = pd.read_parquet(df_path_f10)
    # data_f10 = partition_high_signal(df_f10)
    #
    # logger_h.info(f"Begin x_train")
    # data_f10.x_train.to_parquet(
    #     f"../data/processed/first_ten/x_train.parquet",
    #     compression="gzip",
    # )
    #
    # logger_h.info(f"Begin y_train")
    # data_f10.y_train.to_parquet(
    #     f"../data/processed/first_ten/y_train.parquet",
    #     compression="gzip",
    # )
    #
    # logger_h.info(f"Begin x_test")
    # data_f10.x_test.to_parquet(
    #     f"../data/processed/first_ten/x_test.parquet",
    #     compression="gzip",
    # )
    #
    # logger_h.info(f"Begin y_test")
    # data_f10.y_test.to_parquet(
    #     f"../data/processed/first_ten/y_test.parquet",
    #     compression="gzip",
    # )
    #
    # logger_h.info(f"Begin DF")
    # data_f10.df.to_parquet(
    #     f"../data/processed/first_ten/balanced_data_f10.parquet",
    #     compression="gzip",
    # )



    # even_odd_case = create_temporal_coverage_data(
    #     waveform_dir=Path("../data/dayfiles"),
    #     this_case="even_odd",
    # )
    #
    # even_odd_case.train.to_parquet(
    #     f"../data/processed/even_odd/even_odd_train.parquet",
    #     compression="gzip",
    # )
    #
    # even_odd_case.test.to_parquet(
    #     f"../data/processed/even_odd/even_odd_test.parquet",
    #     compression="gzip",
    # )
    #
    # first_ten_case = create_temporal_coverage_data(
    #     waveform_dir=Path("../data/dayfiles"),
    #     this_case="first_ten",
    # )
    #
    # first_ten_case.train.to_parquet(
    #     f"../data/processed/first_ten/first_ten_train.parquet",
    #     compression="gzip",
    # )
    #
    # first_ten_case.test.to_parquet(
    #     f"../data/processed/first_ten/first_ten_test.parquet",
    #     compression="gzip",
    # )
