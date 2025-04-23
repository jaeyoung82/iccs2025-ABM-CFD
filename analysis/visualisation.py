import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import r2_score
from const import *
from data_processing import process_traj_data


# load npy files, if not found, process traj_data & save to npy files & load them
def get_ps_arrays(ps_model_file, traj_files, output_npy_files):
    # for f in output_npy_files:
    try:
        output_arrays = [np.load(f) for f in output_npy_files]
    except FileNotFoundError:
        print("No npy files found. Processing traj data now...")
        output_arrays = []
        for traj_input, npy_output in zip(traj_files, output_npy_files):
            print("#" * 130)
            print(f"Processing {traj_input}")
            output_arrays.append(
                process_traj_data(ps_model_file, traj_input, npy_output)
            )
            print()
    finally:
        print(f"{len(output_arrays)} npy files loaded")
        return output_arrays
        # for o in output_arrays:
        #     print("#"*100)
        #     print(o.shape)
        #     print("First 5 timesteps", pd.Series(o[:150, 0]).unique())
        #     print("Agent IDs", o[:30, 1])


# (0) consolidate all repetitions of the same experiment just by taking mean across each data point
def get_avg_data(npy_arrays: list[np.ndarray]) -> pd.DataFrame:
    npy_list = []
    for i, arr in enumerate(npy_arrays):
        loaded_df = pd.DataFrame(arr, columns=output_col)
        # remove agent ID 1 / 31 / 61 ...
        remove_ID = loaded_df.loc[0, "id"]
        loaded_df = loaded_df[loaded_df["id"] != remove_ID]
        loaded_df.loc[:, "timeStep"] = loaded_df["timeStep"] / writer_freq
        loaded_df = loaded_df.to_numpy()
        npy_list.append(loaded_df)
    agent_npy_stack = np.stack(npy_list, axis=0)
    agent_data = np.mean(agent_npy_stack, axis=0)
    agent_data = pd.DataFrame(agent_data, columns=output_col).astype({"id": int})
    return agent_data


# (1) consolidate all repetitions of the same experiment by grouping agent ID
#   groupby -> agg -> stack -> mean
def get_agg_agent_data(npy_arrays: list[np.ndarray]) -> pd.DataFrame:
    agent_npy_list = []
    for i, arr in enumerate(npy_arrays):
        loaded_df = pd.DataFrame(arr, columns=output_col)
        agent_df = loaded_df.groupby(["id"], as_index=False).agg(
            {"ps": agent_ps_operator, "eu_dist": "mean"}
        )
        agent_npy = agent_df.drop(0).to_numpy()
        # agent ID increases across repeated simulations, so change them back to 1->30
        agent_npy[:, 0] = agent_npy[:, 0] % num_agents
        agent_npy[-1, 0] = 30
        agent_npy_list.append(agent_npy)

    agent_npy_stack = np.stack(agent_npy_list, axis=0)
    agent_data = np.mean(agent_npy_stack, axis=0)
    agent_data = pd.DataFrame(agent_data, columns=output_col[1:]).astype({"id": int})
    return agent_data


# (2) consolidate all repetitions of the same experiment by grouping timeStep
#   remove emitter -> simplify frames -> groupby -> agg -> stack -> mean
def get_agg_timestep_data(npy_arrays: list[np.ndarray]) -> pd.DataFrame:
    timestep_npy_list = []
    for i, arr in enumerate(npy_arrays):
        loaded_df = pd.DataFrame(arr, columns=output_col)
        # remove agent ID 1 / 31 / 61 ...
        remove_ID = loaded_df.loc[0, "id"]
        loaded_df = loaded_df[loaded_df["id"] != remove_ID]
        # simplify frame difference to be 1
        loaded_df.loc[:, "timeStep"] = loaded_df["timeStep"] / writer_freq
        timestep_df = loaded_df.groupby(["timeStep"], as_index=False).agg(
            {"ps": timestep_ps_operator, "eu_dist": "mean"}
        )
        timestep_npy = timestep_df.to_numpy()
        timestep_npy_list.append(timestep_npy)

    timestep_npy_stack = np.stack(timestep_npy_list, axis=0)
    timestep_data = np.mean(timestep_npy_stack, axis=0)
    timestep_data = pd.DataFrame(
        timestep_data, columns=["timeStep", "ps", "eu_dist"]
    ).astype({"timeStep": int})
    return timestep_data


# (3) for a subset of agents, consolidate all repetitions of the same experiment by grouping timeStep
#   remove emitter -> simplify frames -> *isin* -> groupby -> agg -> stack -> mean
def get_agg_timestep_agent_subset(
    expt_rep_npy_files: list[np.ndarray], group: list[int]
) -> pd.DataFrame:
    timestep_npy_list = []
    for i, arr in enumerate(expt_rep_npy_files):
        # print("Processing arr", i+1)
        # agent ID increases across repeated simulations, so change them back to 1->30
        arr[:, 1] = arr[:, 1] % num_agents
        loaded_df = pd.DataFrame(arr, columns=output_col)
        loaded_df.loc[loaded_df["id"] == 0, "id"] = 30
        # print(loaded_df["id"].unique())
        # agent ID increases across repeated simulations, so increase all IDs in the group by 30
        # apparent_group = [agent_id + i * num_agents for agent_id in group]
        loaded_df = loaded_df[loaded_df["id"].isin(group)]
        # simplify frame difference to be 1
        loaded_df.loc[:, "timeStep"] = loaded_df["timeStep"] / writer_freq
        timestep_df = loaded_df.groupby(["timeStep"], as_index=False).agg(
            {"ps": timestep_ps_operator, "eu_dist": "mean"}
        )
        # print(i, timestep_df.shape)
        timestep_npy = timestep_df.to_numpy()
        timestep_npy_list.append(timestep_npy)

    # print("length: ", len(timestep_npy_list))

    timestep_npy_stack = np.stack(timestep_npy_list, axis=0)
    timestep_subset = np.mean(timestep_npy_stack, axis=0)
    timestep_subset = pd.DataFrame(
        timestep_subset, columns=["timeStep", "ps", "eu_dist"]
    ).astype({"timeStep": int})
    return timestep_subset


def plot_histogram(
    data: pd.Series,
    dp: int,
    num_bins: int,
    title: str = "Histogram",
    save_filepath: str = "",
):

    min_value = math.floor(data.min() * 10**dp) / 10**dp
    max_value = math.ceil(data.max() * 10**dp) / 10**dp
    bin_range = (min_value, max_value)

    plt.hist(data, bins=num_bins, range=bin_range, edgecolor="black")

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title(title)

    if save_filepath:
        plt.savefig(save_filepath, dpi=300)
    else:
        plt.show()


def plot_best_fit_line(
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    title: str = "Best Fit Line",
    save_filepath: str = "",
):
    x, y = df[xlabel], df[ylabel]
    plt.scatter(x, y, marker="x")

    # labels = list(range(2,30))
    # for i, label in enumerate(labels):
    #     plt.text(x[i], y[i], label, fontsize=9, ha='right')

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    best_fit_line = slope * x + intercept

    plt.plot(x, best_fit_line, color="red", label="Best Fit Line")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.suptitle(title)
    # plt.title(f"R2 value: {r_value**2:.5f}")

    plt.grid(True)

    if save_filepath:
        plt.savefig(save_filepath, dpi=300)
    else:
        plt.show()


def plot_best_fit_curve(
    df: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    title: str = "Best Fit Curve",
    save_filepath: str = "",
):
    x, y = df[xlabel], df[ylabel]
    plt.scatter(x, y, marker="x")

    # labels = list(range(2,30))
    # for i, label in enumerate(labels):
    #     plt.text(x[i], y[i], label, fontsize=9, ha='right')

    coefficients = np.polyfit(x, y, 2)
    best_fit_curve = np.poly1d(coefficients)
    x_curve = np.linspace(x.min(), x.max(), 100)
    y_predicted = best_fit_curve(x)
    r_squared = r2_score(y, y_predicted)

    plt.plot(x_curve, best_fit_curve(x_curve), color="red")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(title)
    plt.title(f"R2 value: {r_squared:.5f}")

    plt.grid(True)

    if save_filepath:
        plt.savefig(save_filepath, dpi=300)
    else:
        plt.show()


# credits: https://stackoverflow.com/questions/7761778/matplotlib-adding-second-axes-with-transparent-background


def plot_time_series(
    df: pd.DataFrame, title: str, suptitle: str = "", save_filepath: str = ""
):

    num_entry = len(df)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # plot ps on primary y-axis
    ax1.plot(df["timeStep"], df["ps"], label="ps", color="blue")
    ax1.set_xlabel("Minutes")
    ax1.set_ylabel("ps", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # first x axis for frames
    # xticks = range(0, num_entry, num_entry * writer_freq // 60)
    # ax1.set_xticks(xticks)
    # ax1.set_xticklabels(xticks)
    # xticks = range(0, 60 // writer_freq + 1, 1)
    ax1.set_xticks(range(0, num_entry, num_entry * writer_freq // 60))
    ax1.set_xticklabels(range(0, 60 // writer_freq + 1, 1))

    # plot eu_dist on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(df["timeStep"], df["eu_dist"], label="eu_dist", color="green")
    ax2.set_ylabel("eu_dist", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # second x axis for minutes
    # ax3 = ax1.twiny()
    # fig.subplots_adjust(bottom=0.1)
    # ax3.set_frame_on(True)
    # ax3.patch.set_visible(False)
    # ax3.xaxis.set_ticks_position("bottom")
    # ax3.xaxis.set_label_position("bottom")
    # ax3.spines["bottom"].set_position(("outward", 40))

    # ax3.set_xticks(ax1.get_xticks())
    # ax3.set_xticklabels(range(0, 60 // writer_freq + 1, 1))
    # ax3.set_xlim(ax1.get_xlim())
    # ax3.set_xlabel("Minutes")

    if suptitle:
        plt.suptitle(suptitle)
    plt.title(title)

    ax1.grid(True)

    if save_filepath:
        plt.savefig(save_filepath, dpi=300)
    else:
        plt.show()


def plot_time_series_groups(
    df_list: list[pd.DataFrame], title: str, suptitle: str = "", save_filepath: str = ""
):
    targets = ("restroom", "subgroup", "table")
    num_entry = len(df_list[0])
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # plot ps on y-axis
    for i, df in enumerate(df_list):
        ax1.plot(df["timeStep"], df["ps"], label=targets[i])
    ax1.set_xlabel("Minutes")
    ax1.set_ylabel("ps", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # x axis for minutes
    ax1.set_xticks(range(0, num_entry, num_entry * writer_freq // 60))
    ax1.set_xticklabels(range(0, 60 // writer_freq + 1, 1))

    if suptitle:
        plt.suptitle(suptitle)
    plt.title(title)

    plt.legend()
    plt.grid(True)

    if save_filepath:
        plt.savefig(save_filepath, dpi=300)
    else:
        plt.show()


def plot_time_series_multi(df_list: list[pd.DataFrame]):
    expt_name = ("R", "G", "T")
    expt_color = ("b", "g", "r")
    num_entry = len(df_list[0])
    fig, ax = plt.subplots(figsize=(10, 6))

    # plot ps on y-axis
    for i, df in enumerate(df_list):
        ax.plot(df["timeStep"], df["ps"], label=expt_name[i], color=expt_color[i])
    ax.set_xlabel("minutes")
    ax.set_ylabel("ps")
    ax.tick_params(axis="y")

    # x axis for minutes
    ax.set_xticks(range(0, num_entry, num_entry * writer_freq // 60))
    ax.set_xticklabels(range(0, 60 // writer_freq + 1, 1))

    plt.legend()
    plt.grid(True)
    plt.show()