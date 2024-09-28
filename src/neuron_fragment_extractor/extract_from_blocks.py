"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Given a whole brain predicted segmentation in the form of a directory of swc
files and a set of target swc files, this code extracts predicted swc files
that have a node within some range of a node from a target swc files.

"""

from copy import deepcopy as cp
from random import sample
from time import time
from tqdm import tqdm

import os
import json
import numpy as np
import shutil

from deep_neurographs.utils import swc_util, util
from deep_neurographs.utils.img_util import get_chunk_labels


def extract_preds():
    # Read predicted swc files from cloud
    print("Loading Fragments...")
    reader = swc_util.Reader(anisotropy, min_size)
    swc_dicts = reader.load(fragments_pointer)
    print("# Fragments:", len(swc_dicts))

    # Main
    print("\nGenerating Neuron Reconstruction Datasets...")
    for block_id in util.list_subdirs(target_swc_dir, keyword="block"):
        # Initializations
        labels = get_labels(block_id)
        util.mkdir(os.path.join(pred_swc_dir, block_id))

        # Parse swc files
        cnt = 0
        for swc_dict in tqdm(swc_dicts, desc=block_id):
            swc_id = int(swc_dict["swc_id"])
            if swc_id in labels:
                path = f"{pred_swc_dir}/{block_id}/{swc_id}.swc"
                graph, _ = swc_util.to_graph(swc_dict, set_attrs=True)
                swc_util.write(path, graph)
                cnt += 1
        print("# Fragments Found:", cnt)
        print("")


def get_labels(block_id):
    labels_path = f"{gcs_path}/label_mask"
    metadata_path = f"{target_swc_dir}/{block_id}/metadata.json"
    origin, shape = util.read_metadata(metadata_path)
    return get_chunk_labels(labels_path, origin, shape, from_center=False)


if __name__ == "__main__":
    # Parameters
    bucket_name = "allen-nd-goog"
    dataset = "706301"
    pred_id = "202405_106997260_633_mean100_dynamic"

    anisotropy = [0.748, 0.748, 1.0]
    min_size = 20

    # Initialize paths
    root_dir = f"/home/jupyter/workspace/data/{dataset}"
    gcs_path = f"from_google/whole_brain/{dataset}/{pred_id}"
    target_swc_dir = f"{root_dir}/target_swcs/blocks"
    pred_swc_dir = f"{root_dir}/pred_swcs/{pred_id}"
    util.mkdir(pred_swc_dir, delete=True)

    fragments_pointer = {
        "bucket_name": "allen-nd-goog",
        "path": os.path.join(gcs_path, "swcs"),
    }
    print(f"from_google/whole_brain/{dataset}/{pred_id}/swcs")

    # Run extraction
    t0 = time()
    extract_preds()
    t, unit = util.time_writer(time() - t0)
    print("\nRuntime: {} {}\n".format(t, unit))
