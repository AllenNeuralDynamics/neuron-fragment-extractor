"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Given a whole brain predicted segmentation in the form of a directory of swc
files and a set of target swc files, this code extracts predicted swc files
that have a node within some range of a node from a target swc files.

"""

import os
import json
import multiprocessing
import numpy as np
import shutil
from deep_neurographs import intake, swc_utils, utils
from deep_neurographs.machine_learning import inference
from random import sample
from scipy.spatial import KDTree
from time import time


def extract_swcs():
    # Initializations
    print("Downloading SWC Files...")
    kdtrees = build_kdtrees()
    swc_dicts = intake.download_gcs_zips(
        bucket_name, cloud_path, min_size, [0.748, 0.748, 1.0]
    )

    print("\nSWC Overview...")
    print("# target swcs:", len(kdtrees))
    print("# predicted swcs:", utils.reformat_number(len(swc_dicts)))

    # Create shared data structures
    print("\nCreating shared data structures")
    manager = multiprocessing.Manager()
    shared_swc_dicts = manager.list(swc_dicts)
    del swc_dicts

    # Assign processes
    print("Assigning procesesses")
    processes = []
    process_id = 0
    for target_id, kdtree in kdtrees.items():
        process_id += 1
        process = multiprocessing.Process(
            target=query_kdtree,
            args=(kdtree, shared_swc_dicts, target_id, process_id))
        processes.append(process)
        process.start()

    print("Querying KD-Trees")
    for process in processes:
        process.join()


def query_kdtree(kdtree, swc_dicts, target_id, process_id):
    cnt = 1
    chunk_size = len(swc_dicts) * 0.05
    t0, t1 = utils.init_timers()
    for i, swc_dict in enumerate(swc_dicts):
        # Check whether component is close
        for xyz in swc_dict["xyz"][::5]:
            d, _ = kdtree.query(xyz, k=1)
            if d < search_radius:
                swc_id = swc_dict["swc_id"]
                path = f"{pred_swc_dir}/{target_id}/{swc_id}.swc"
                swc_utils.write(path, swc_dict)
                break

        # Report progress
        if i >= cnt * chunk_size:
            percent = int(100 * i / len(swc_dicts))
            print(f"Process {process_id} completed {percent}% of tasks")
            cnt += 1


def build_kdtrees():
    # Initializations
    kdtrees = dict()
    paths = utils.list_paths(target_swc_dir, ext=".swc")
    swc_dicts, _ = intake.process_local_paths(paths, anisotropy=[0.748, 0.748, 1.0])

    # Main
    for swc_dict in swc_dicts:
        swc_id = swc_dict["swc_id"]
        kdtrees[swc_id] = KDTree(swc_dict["xyz"][::5])
        utils.mkdir(os.path.join(pred_swc_dir, swc_id))
    return kdtrees        


if __name__ == "__main__":
    # Parameters
    bucket_name = "allen-nd-goog"
    dataset = "706301"
    pred_id = "202405_106997260_633_mean100_dynamic"
    min_size = 25
    search_radius = 100

    # Initialize paths
    cloud_path = f"from_google/whole_brain/{dataset}/{pred_id}/swcs"
    root_dir = f"/home/jupyter/workspace/data/{dataset}"
    target_swc_dir = f"{root_dir}/target_swcs"
    pred_swc_dir = f"{root_dir}/pred_swcs/{pred_id}"
    utils.mkdir(pred_swc_dir, delete=True)

    # Run extraction
    t0 = time()
    extract_swcs()
    t, unit = utils.time_writer(time() - t0)
    print("\nextract_swcs(): {} {}".format(t, unit))
