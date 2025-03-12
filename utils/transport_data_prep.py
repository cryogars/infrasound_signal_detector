import sys
sys.path.append('../')

import gc
import pathlib
from collections import namedtuple
from typing import Union, NamedTuple

# Base Libraries : Plotting | Data Wrangling | Scientific Data Manipulation
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# System utility libraries
import logging
import os
from pathlib import Path

from utils import config
from utils.helpers import get_logger
logger = get_logger(__file__)


def prep_other_sensor(the_station, the_sensor) -> None:
    this_happy_path = f"{config.PROCESSED_DATA}/two_high_signal"
    path = f"{this_happy_path}/{the_station}/full_waveform_stats_df_{the_station}_test_sensor_{the_sensor}.parquet"
    df = pd.read_parquet(path)

    logger.debug(
        f"Count of label name for Station: {the_station}, Sensor: {the_sensor} \n{df['label_names'].value_counts()}"
    )
    logger.debug(
        f"Count of label  for Station: {the_station}, Sensor: {the_sensor} \n{df['label'].value_counts()}"
    )

    high_signal_class = df[df["label_names"] == "high_signal"]
    sampled_list = [high_signal_class]

    class_labels = list(df["label_names"].unique())
    class_labels.remove("high_signal")

    for class_label in class_labels:
        this_class_size = df["label_names"].value_counts().get(class_label)
        this_class_size = int(this_class_size / 4)
        class_subset = df[df["label_names"] == class_label]
        sampled_label = resample(
            class_subset, replace=False, n_samples=this_class_size, random_state=42
        )
        sampled_list.append(sampled_label)
    balanced_data = pd.concat(sampled_list)

    logger.debug(f"Convert to two label")
    balanced_data["label"] = balanced_data["label_names"].apply(lambda x: 1 if x == "high_signal" else 0)
    balanced_data["label_names"] = balanced_data["label_names"].apply(
        lambda x: x if x == "high_signal" else "noise"
    )
    logger.debug(f"Completed convert to two label")

    logger.debug(
        f"Count of label name for Station: {the_station}, Sensor: {the_sensor} \n{balanced_data['label_names'].value_counts()}"
    )
    logger.debug(
        f"Count of label  for Station: {the_station}, Sensor: {the_sensor} \n{balanced_data['label'].value_counts()}"
    )

    #
    x = balanced_data.drop(columns=["const", "samples", "xcth", "label", "label_names"])
    y = balanced_data[["label"]]
    #
    logger.debug(f"Saving X and Y to Parquet")

    x_path = f"{this_happy_path}/{the_station}/X_sensor{the_sensor}_full.parquet"
    y_path = f"{this_happy_path}/{the_station}/y_sensor{the_sensor}_full.parquet"

    x.to_parquet(path=x_path, compression="gzip")
    y.to_parquet(path=y_path, compression="gzip")
    logger.debug(f"Completed saving X and Y to Parquet")


def create_test(df: pd.DataFrame) -> namedtuple:
    """
    :param df: Pandas Dataframe of Labeled Inputs
    :return: named tuple of the partition set
    """
    Result = namedtuple(
        typename="Result", field_names=["df", "x_test", "y_test"]
    )
    # -----------------------------------------------
    high_signal_class = df[df["label_names"] == "high_signal"]
    sampled_list = [high_signal_class]

    class_labels = list(df["label_names"].unique())
    class_labels.remove("high_signal")

    for class_label in class_labels:
        this_class_size = df["label_names"].value_counts().get(class_label)
        this_class_size = int(this_class_size/4)
        class_subset = df[df["label_names"] == class_label]
        sampled_label = resample(
            class_subset, replace=False, n_samples=this_class_size, random_state=42
        )
        sampled_list.append(sampled_label)
    balanced_data = pd.concat(sampled_list)
    # --------------------------------------------------------
    balanced_data["label"] = balanced_data["label_names"].apply(
        lambda label: 1 if label == "high_signal" else 0
    )

    balanced_data["label_names"] = balanced_data["label_names"].apply(
        lambda label: label if label == "high_signal" else "noise"
    )

    x = balanced_data.drop(columns=["const", "samples", "xcth", "label", "label_names"])
    y = balanced_data[["label"]]
    gc.collect()
    return Result(
        df=balanced_data, x_test=x, y_test=y
    )

if __name__ == "__main__":
    # prep_other_sensor("LCC2", 3)
    ############################################################################################
    ## TRANSPORTABILITY DATA FOR SENSOR AGNOSTIC MODEL
    ############################################################################################
    stations = ["LCC2", "LCC3"]
    sensors = [1, 2, 3]
    for station in stations:
        for sensor in sensors:
            if not (station == "LCC2" and sensor == 3):
                prep_other_sensor(station, sensor)

    ############################################################################################
    ## TRANSPORTABILITY DATA FOR TEMPORAL MODEL
    ############################################################################################

    # logger.info(f"Begin First Ten")
    # df_path_f10 = Path(f"{config.PROCESSED_DATA}/first_ten/first_ten_test.parquet")
    # df_f10 = pd.read_parquet(df_path_f10)
    # data_f10 = create_test(df_f10)
    #
    # logger_h.info(f"Begin x_test")
    # data_f10.x_test.to_parquet(
    #     f"{config.PROCESSED_DATA}/first_ten/x_true_test_ds.parquet",
    #     compression="gzip",
    # )
    #
    # logger.info(f"Begin y_test")
    # data_f10.y_test.to_parquet(
    #     f"{config.PROCESSED_DATA}/first_ten/y_true_test_ds.parquet",
    #     compression="gzip",
    # )

    ############################################################################################
    ## TRANSPORTABILITY DATA FOR SENSOR METEOROLOGICAL MODEL
    ############################################################################################

    # logger.info(f"Begin Even Odd")
    # df_path = Path(f"{config.PROCESSED_DATA}/even_odd/even_odd_test.parquet")
    # df = pd.read_parquet(df_path)
    # data = create_test(df)
    #
    # logger_h.info(f"Begin x_test")
    # data.x_test.to_parquet(
    #     f"{config.PROCESSED_DATA}/even_odd/x_true_test_ds.parquet",
    #     compression="gzip",
    # )
    #
    # logger_h.info(f"Begin y_test")
    # data.y_test.to_parquet(
    #     f"{config.PROCESSED_DATA}/even_odd/y_true_test_ds.parquet",
    #     compression="gzip",
    # )
