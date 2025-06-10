"""
Script to explore data
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

# System libraries
import sys
sys.path.append("../")
from pathlib import Path

# Scientific and Data wrangling libraries
import numpy as np
import pandas as pd

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# User Define libraries
import utils.config as config
from utils.helpers import get_logger

logger = get_logger(__file__)

if __name__ == "__main__":
    data_path = Path(
        f"{config.PARENT_DIRECTORY}/data/sample/waveform_stats_df_hist.parquet"
    )
    waveform_df = pd.read_parquet(data_path)
    print(waveform_df['label_names'].unique())

    data_path3 = Path(
        f"{config.PARENT_DIRECTORY}/data/sample/waveform_stats_df_histl3.parquet"
    )
    waveform_df3 = pd.read_parquet(data_path3)


    #############################################################
    # Plotting 3D
    #############################################################
    # hqs = waveform_df[waveform_df['label_names'].isin(['high_signal','ambiguous','noise'])]
    hqs = waveform_df[['const','xcth','label','label_names']]
    hqs_count = hqs["label_names"].value_counts().get('high_signal',0)
    sig_count = hqs["label_names"].value_counts().get('signal',0)
    noise_count = hqs["label_names"].value_counts().get('noise',0)
    amb_count = hqs["label_names"].value_counts().get('ambiguous',0)

    nhqs_count = sig_count + noise_count + amb_count


    logger.info(f"Begin 3d Plotting")
    consistency = hqs["const"]
    correlation = hqs["xcth"]
    #
    # # binning
    consistency_bins = np.linspace(start=0, stop=150, num=30)
    correlation_bins = np.linspace(start=0, stop=1, num=50)
    #
    # # create histogram
    the_histogram, consistency_edges, correlation_edges = np.histogram2d(
        x=consistency, y=correlation, bins=[consistency_bins, correlation_bins]
    )
    #
    # # Bars coordinates
    consistency_pos, correlation_pos = np.meshgrid(
        consistency_edges[:-1], correlation_edges[:-1], indexing="ij"
    )
    consistency_pos = consistency_pos.ravel()
    correlation_pos = correlation_pos.ravel()
    count_pos = np.zeros_like(consistency_pos)

    dx = np.diff(consistency_edges)[0]
    dy = np.diff(correlation_edges)[0]
    dz = the_histogram.ravel()

    # non_zero = count_pos > 0
    # consistency_pos = consistency_pos[non_zero]
    # correlation_pos = correlation_pos[non_zero]
    # count_pos = count_pos[non_zero]
    # dz = dz[non_zero]
    #
    # # Color mapping
    colors = np.where(
        (consistency_pos <= 5) & (correlation_pos >= 0.8),
        "yellow",
        np.where(
            (consistency_pos <= 5) & (correlation_pos < 0.8) & (correlation_pos > 0.1),
            "orange", # orange
            np.where(
                (consistency_pos <= 25)
                & (correlation_pos > 0.1)
                & (correlation_pos <= 0.8),
                "blue", # blue
                np.where(
                    (consistency_pos >= 25)
                    & (correlation_pos > 0.1)
                    & (correlation_pos < 0.6),
                    "purple", # purple
                    "white",
                ),
            ),
        ),
    )


    #
    legend_elements = [
        Patch(facecolor="yellow", label=f"High Signal: {hqs_count}"),
        Patch(facecolor="orange", label=f"Signal: {sig_count} "),
        Patch(facecolor="purple", label=f"Noise: {noise_count} "),
        Patch(facecolor="blue", label=f"Ambiguous: {amb_count} "),
    ]

    # legend_elements = [
    #     Patch(facecolor="yellow", label=f"HQS: {hqs_count}"),
    #     Patch(facecolor="blue", label=f"NHQS: {nhqs_count} "),
    # ]
    #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.bar3d(consistency_pos, correlation_pos, count_pos, dx, dy, dz, color=colors)


    ax.set_xlabel("Consistency", fontsize=16, weight="bold")
    ax.set_ylabel("Cross Correlation", fontsize=16, weight="bold")
    ax.set_zlabel("Count", fontsize=16, weight="bold")
    ax.set_title("B: LCC3", fontsize=20, weight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.legend(handles=legend_elements, loc="upper right",fontsize=16)
    plt.savefig(f'{config.PARENT_DIRECTORY}/plots/hist3d_LCC34.jpg',dpi=300)
    logger.info(f"End 3d Plotting")

    #############################################################
    # Plotting Bar Chart
    #############################################################

    # logger.info(f"Plotting Class Count")
    # plt.figure()
    # class_count = waveform_df["label_names"].value_counts()
    # # class_count.plot(kind='bar')
    # print(class_count)
    # print(sum(class_count))
    #
    # sns.countplot(x="label_names", data=waveform_df)
    # plt.xlabel("labels")
    # plt.ylabel("counts")
    # plt.title("Bar Chart of Labels Counts")
    # plt.savefig(
    #     Path(f"{config.PARENT_DIRECTORY}/plots/label_count.png"), bbox_inches="tight"
    # )
    # logger.info(f"Completed Class Count Plot")

    # plt.show()


    # Create 2D histogram bins
    # correlation = hqs.xcth  # x
    # samples = hqs.const  # y
    #
    # print(np.max(correlation))
    # print(np.min(correlation))
    #
    # print(np.max(samples))
    # print(np.min(samples))
    #
    # hist, xedges, yedges = np.histogram2d(correlation, samples, bins=[30, 15], range=[[0, 1], [0, 150]])
    #
    # # Prepare for 3D bar chart plotting
    # xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    # xpos = xpos.ravel()
    # ypos = ypos.ravel()
    # zpos = np.zeros_like(xpos)
    # # dx = dy = 0.03  # Bin width
    # # dz = hist.ravel()
    # dx = np.diff(yedges)[0]
    # dy = np.diff(xedges)[0]
    # dz = hist.ravel()
    #
    # # Color mapping
    # colors = np.where(
    #     (ypos <= 5) & (xpos >= 0.8),
    #     "yellow",
    #     np.where(
    #         (ypos <= 5) & (xpos < 0.8),
    #         "orange",
    #         np.where(
    #             (ypos <= 25) & (ypos > 5) & (xpos <= 0.8),"blue",
    #             "black",
    #         ),
    #     ),
    # )
    #
    # # Create 3D bar chart
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, zsort='average',color=colors)
    #
    # ax.set_xlabel('Correlation', fontsize=17)
    # ax.set_ylabel('Samples', fontsize=17)
    # ax.set_zlabel('Count', fontsize=17)
    # ax.set_title('3D Histogram of Correlation vs Samples', fontsize=18)
    # plt.tight_layout()
    # plt.savefig(f'{config.PARENT_DIRECTORY}/plots/hist2d.jpg', dpi =300)