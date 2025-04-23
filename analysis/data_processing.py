import pickle
import numpy as np
import pandas as pd
from const import *

"""
AGENTS ARE NOT IN CIRCULAR ORDER

            17  ...  30
        ┌──────────────────┐
    1   │                  │   16
        └──────────────────┘
             2  ...  15

    simulation model dimensions:
    X: 0 to 23.65 m
    Y: 0 to 12.05 m

    infection model dimensions:
    X: -11.65 to 12 m
    Y: -4.5 to 7.55 m
"""


def get_gridpoints(traj_batch):
    # (1) calculate gridpoints coordinates
    n_coord = (
        n_coors * 2 + 1
    )  # number of gridpoints in both horizontal and vertical directions.
    # (x_min, y_min) = bottom left coordinates of the grid
    x_min = 0.5 * length_grid  # (half of length_grid) to the right of agent
    y_min = -n_coors * length_grid  # (n_coors*length_grid) downwards of agent

    points_xy = []  # list to hold gridpoints coordinates
    for i in range(0, n_coord):
        for j in range(0, n_coord):
            point_x = x_min + length_grid * i
            point_y = y_min + length_grid * j
            points_xy.append([round(point_x, 4), round(point_y, 4)])

    # (2) calculate angle of rotation
    heading_x = np.array(traj_batch[:, 2])
    heading_y = np.array(traj_batch[:, 3])
    theta = np.arctan2(heading_y, heading_x)

    # (3) rotation from (1,0) heading
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    rotated_grid_xy = np.dot(np.array(points_xy), R.T)
    rotated_grid_xy += np.array(traj_batch[:, :2])
    output_arr = np.transpose(rotated_grid_xy, (1, 0, 2)).reshape(
        (n_total_gridpoints, 2)
    )

    return output_arr


def compute_ps(point_batch, source_xy, xgb):
    source_batch = np.full((n_total_gridpoints, 2), source_xy + translation_vector)
    src_pt_batch = np.c_[source_batch + translation_vector, point_batch]
    eudist_batch = np.linalg.norm(
        src_pt_batch[:, :2] - src_pt_batch[:, 2:], axis=1, keepdims=True
    )
    X_batch = np.c_[src_pt_batch, eudist_batch]
    y_pred_batch_log = xgb.predict(X_batch)
    ps_batch = 10**y_pred_batch_log
    return ps_batch


def aggregate_ps(ps_batch):
    ps_eudist_grid_batch = ps_batch.reshape((n_receivers, n_gridpoints_per_agent, 1))
    if ps_operator == "sum":
        ps_agg_col = np.sum(ps_eudist_grid_batch[:, :, 0], axis=1)
    elif ps_operator in ("avg", "mean"):
        ps_agg_col = np.mean(ps_eudist_grid_batch[:, :, 0], axis=1)
    else:
        raise ValueError(
            f"ps_operator must be either 'sum' or 'avg', not '{ps_operator}'."
        )
    return ps_agg_col


def compute_agent_eudist(receiver_batch, source_xy):
    source_batch = np.full((n_receivers, 2), source_xy)
    src_rcv_batch = np.c_[source_batch, receiver_batch]
    eudist_batch = np.linalg.norm(
        src_rcv_batch[:, :2] - src_rcv_batch[:, 2:], axis=1, keepdims=True
    )
    return eudist_batch


def process_traj_data(
    ps_model_filepath: str, traj_filepath: str, output_npy_filepath: str = ""
):
    """
    Calculates passive scalar and euclidean distance from a trajectory data file.
    NDArray output will be saved to the npy file and also returned.

    Args:
        traj_filepath (str): Filepath to trajectory data from project root, must end with '.csv'
        ps_model_filepath (str): Filepath to passive scalar model data from project root, must end with '.p'
        output_npy_filepath (str): Filepath to create npy output file, must end with '.npy'

    Returns:
        output_arr: NDArray[float64]
    """
    # load data files
    xgb = pickle.load(open(ps_model_filepath, "rb"))
    input_arr = np.genfromtxt(traj_filepath, delimiter=";", skip_header=1)
    # only keep the first 6 columns (frame, id, x, y, xHeading, yHeading)
    input_arr = input_arr[:, :6]
    # check input data dimensions
    input_shape = ((n_frames + 1) * n_agents, 6)
    if input_arr.shape != input_shape:
        raise ValueError(
            f"input_arr should have shape {input_shape} instead of {input_arr.shape}"
        )

    # prepare input & output arrays

    # sort input_arr by timeStep then id
    sorted_idx = np.lexsort((input_arr[:, 1], input_arr[:, 0]))
    sorted_input_arr = input_arr[sorted_idx]

    # initialise output_arr with timeStep and id values from input_arr
    output_arr = np.empty((input_shape[0], 4))  # 4 columns (timeStep, id, ps, eu_dist)
    output_arr[:, :2] = sorted_input_arr[:, :2]  # copy timeStep & id columns

    # processing
    print(f"Frames to process: {n_frames}\n")
    for f in range(first_frame, n_frames + 1):

        if f % 500 == 0:
            print(f"Frames processed: {f}")

        source_xy = sorted_input_arr[f * n_agents, 2:4]
        traj_batch = sorted_input_arr[f * n_agents + 1 : (f + 1) * n_agents, 2:]

        # (1) calculate xy pos of every receiver's gridpoints
        if traj_batch.shape != (n_receivers, 4):
            raise ValueError(
                f"traj_batch should have shape ({n_receivers},4) instead of {traj_batch.shape}"
            )
        rotated_pos_batch = get_gridpoints(traj_batch)

        # (2) calculate ps from the gridpoints
        if rotated_pos_batch.shape != (n_total_gridpoints, 2):
            raise ValueError(
                f"point_batch should have shape ({n_total_gridpoints},2) instead of {rotated_pos_batch.shape}"
            )
        if source_xy.shape != (2,):
            raise ValueError(
                f"source_xy should have shape (2,) instead of {source_xy.shape}"
            )
        ps_batch = compute_ps(rotated_pos_batch, source_xy, xgb)

        # (3) apply some operator across ps
        if ps_batch.shape != (n_total_gridpoints,):
            raise ValueError(
                f"ps_eudist_batch should have shape ({n_total_gridpoints},1) instead of {ps_batch.shape}"
            )
        agg_ps_batch = aggregate_ps(ps_batch)

        # (4) calculate eudist between receiver and emitter agents
        receiver_batch = traj_batch[:, :2]
        eudist_batch = compute_agent_eudist(receiver_batch, source_xy)

        # (5) write ps & eudist to output_arr
        result_batch = np.c_[agg_ps_batch, eudist_batch]
        if result_batch.shape != (n_receivers, 2):
            raise ValueError(
                f"result_batch should have shape ({n_receivers},2) instead of {result_batch.shape}"
            )
        output_arr[f * n_agents, 2:] = 0
        output_arr[f * n_agents + 1 : (f + 1) * n_agents, 2:] = result_batch

    print("\nFinished processing.")

    # save raw output to npy file if specified
    if output_npy_filepath:
        np.save(output_npy_filepath, output_arr)
        print(f"Output saved to: {output_npy_filepath}")

    return output_arr
