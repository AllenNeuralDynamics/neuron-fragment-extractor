"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


This code extracts fragments from a predicted segmentation that are "close" to
at least one nearon from a set of whole-brain ground truth tracings. A
fragment is said to be close if there exists a point within 50um from a ground
truth tracing.

"""

import multiprocessing
import os
from time import time

from deep_neurographs.utils import swc_util, util
from scipy.spatial import KDTree


def extract_swcs():
    # Initializations
    print("Downloading SWC Files...")
    reader = swc_util.Reader(anisotropy, min_size)
    kdtrees = build_kdtrees(reader)
    swc_dicts = reader.load(swc_pointer)

    print("\nSWC Overview...")
    print("# target swcs:", len(kdtrees))
    print("# predicted swcs:", util.reformat_number(len(swc_dicts)))

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
            args=(kdtree, shared_swc_dicts, target_id, process_id),
        )
        processes.append(process)
        process.start()

    print("Querying KD-Trees")
    for process in processes:
        process.join()


def query_kdtree(kdtree, swc_dicts, target_id, process_id):
    cnt = 1
    chunk_size = len(swc_dicts) * 0.05
    for i, swc_dict in enumerate(swc_dicts):
        # Check whether component is close
        for xyz in swc_dict["xyz"][::5]:
            d, _ = kdtree.query(xyz, k=1)
            if d < search_radius:
                swc_id = swc_dict["swc_id"]
                path = f"{pred_swc_dir}/{target_id}/{swc_id}.swc"
                swc_util.write(path, swc_dict)
                break

        # Report progress
        if i >= cnt * chunk_size:
            percent = int(100 * i / len(swc_dicts))
            print(f"Process {process_id} completed {percent}% of tasks")
            cnt += 1


def build_kdtrees(reader):
    kdtrees = dict()
    paths = util.list_paths(target_swc_dir, extension=".swc")
    for swc_dict in reader.load(paths):
        swc_id = swc_dict["swc_id"]
        kdtrees[swc_id] = KDTree(swc_dict["xyz"][::5])
        util.mkdir(os.path.join(pred_swc_dir, swc_id))
    return kdtrees


if __name__ == "__main__":
    # Parameters
    bucket_name = "allen-nd-goog"
    dataset = "706301"
    pred_id = "202405_106997260_633_mean100_dynamic"
    anisotropy = [0.748, 0.748, 1.0]
    min_size = 40
    search_radius = 50

    # Local paths
    root_dir = f"/home/jupyter/workspace/graphtrace_data/train/{dataset}"
    target_swc_dir = f"{root_dir}/target_swcs/whole_brain"
    pred_swc_dir = f"{root_dir}/pred_swcs/{pred_id}/"
    util.mkdir(pred_swc_dir)

    pred_swc_dir += "whole_brain"
    util.mkdir(pred_swc_dir)

    # Cloud path
    swc_pointer = {
        "bucket_name": bucket_name,
        "path": f"from_google/whole_brain/{dataset}/{pred_id}/swcs",
    }

    # Run extraction
    t0 = time()
    extract_swcs()
    t, unit = util.time_writer(time() - t0)
    print("\nRuntime: {} {}".format(t, unit))
