import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(dir, env_name):
    figure, axis = plt.subplots(1, sharex=False)
    figure.set_size_inches(8, 4)
    figure.set_dpi(180)
    figure.tight_layout(pad=3.0)

    df_train = pd.read_csv(dir + "log_train.txt")
    df_test = pd.read_csv(dir + "log_test.txt")

    i = np.array(df_train["i"])
    rew_min_train = np.array(df_train["rew_min"])
    rew_avg_train = np.array(df_train["rew_avg"])
    rew_max_train = np.array(df_train["rew_max"])
    axis.plot(i, rew_avg_train, "-", color="#4F68B1", label="Training population")
    axis.fill_between(i, rew_min_train, rew_max_train, color="#4F68B1", alpha=0.2)

    i = np.array(df_test["i"])
    rew_avg_test = np.array(df_test["rew_avg"])
    axis.plot(i, rew_avg_test, "-", color="#CC6669", label="Test")

    axis.set_ylabel("Average Episode Reward")
    axis.set_xlabel("Iterations")
    axis.legend(loc="upper left")
    axis.set_title(env_name)
    axis.set_facecolor("#eaeaf2")
    [axis.spines[side].set_visible(False) for side in axis.spines]
    axis.grid(which="major", color="white", linewidth=1.0)
    axis.set_axisbelow(True)

    plt.savefig(dir + "reward.png")
    plt.close(figure)
