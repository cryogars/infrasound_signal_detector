import numpy as np
import scipy

import matplotlib.pyplot as plt

from utils import config


def heli_coder(
    trace: np.array,
    sample_rate: int = 100,
    scaling: float = None,
    color: str = "k",
) -> None:
    """

    :param sample_rate:
    :param trace: 24hr Infrasound record
    :param sample_rate: either 1: Sample rate in hertz, 2: time in minutes or
                                3: vector of time in julian day
    :param scaling: force the line spacing to equal this value (often in Pa);
                    default is 10 x the RMS amplitude
    :param color:
    :return: scale_bar
            h:
    """

    HOURS_IN_A_DAY = 24
    scaling = scaling if scaling is not None else np.sqrt(np.mean(trace**2)) * 10

    data_per_minute = sample_rate * 60
    trace = trace / scaling
    reshaped_data = trace.reshape(HOURS_IN_A_DAY, data_per_minute * 60)
    fig, ax = plt.subplots(figsize=(12, 8))
    print(reshaped_data.shape[1])

    for i in range(HOURS_IN_A_DAY):
        x_vals = np.linspace(start=0, stop=60, num=reshaped_data.shape[1])
        ax.plot(
            x_vals,
            reshaped_data[i] * -1 + i,
            label=f"Hour {i + 1}",
            linewidth=0.5,
            color=color,
        )

    # ax.scatter(x=15.004, y=23, color="r", s=100)
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Hour (Stacked)")
    ax.set_title("HeliCoder for Infrasound Waveform")
    ax.set_yticks(np.arange(0, 24, 1))
    ax.set_yticklabels([f"{i}:00" for i in range(24)])
    plt.savefig("../plots/helicoder/heli_coder_p.png", dpi=300)
    plt.show()

    return


if __name__ == "__main__":
    data = scipy.io.loadmat(f"{config.DATA_PATH_DAYFILES}/081data2023Mar22.mat")
    the_trace = data["acfilts"][:, 2, 1]  # [:, config.THIS_SENSOR, config.THIS_STATION]
    heli_coder(trace=the_trace, sample_rate=100, scaling=2, color="k")
