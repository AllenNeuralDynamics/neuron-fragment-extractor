"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Given a whole brain predicted segmentation in the form of a directory of swc
files and a set of target swc files, this code extracts predicted swc files
that have a node within some range of a node from a target swc files.

"""

import os
from time import time

import fastremap
from deep_neurographs.utils import img_util, swc_util, util
from tqdm import tqdm


def extract_preds():
    # Read predicted swc files from cloud
    print("Loading Fragments...")
    swc_reader = swc_util.Reader(anisotropy, min_size)
    swc_dicts = swc_reader.load(fragments_pointer)
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
    # Initializations
    labels_path = f"{gcs_path}/label_mask"
    metadata_path = f"{target_swc_dir}/{block_id}/metadata.json"
    origin, shape = util.read_metadata(metadata_path)

    # Open image
    img = img_util.open_tensorstore(labels_path)
    img = img_util.read_tensorstore(img, origin, shape, from_center=False)
    return set(fastremap.unique(img).astype(int))


if __name__ == "__main__":
    # Parameters
    bucket_name = "allen-nd-goog"
    dataset = "653980"
    pred_id = "202412_73227862_855_mean80_mask40_dynamic"

    anisotropy = (0.748, 0.748, 1.0)
    min_size = 30

    # Initialize paths
    root_dir = f"/home/jupyter/workspace/data/graphtrace/train/{dataset}"
    gcs_path = f"from_google/whole_brain/{dataset}_b0/{pred_id}"
    target_swc_dir = f"{root_dir}/target_swcs/blocks"
    pred_swc_dir = f"{root_dir}/pred_swcs/{pred_id}/block_000"
    util.mkdir(pred_swc_dir, delete=True)

    fragments_pointer = {
        "bucket_name": "allen-nd-goog",
        "path": os.path.join(gcs_path, "swcs_4is_10kic"),
    }

    # Run extraction
    t0 = time()
    extract_preds()
    t, unit = util.time_writer(time() - t0)
    print("\nRuntime: {} {}\n".format(t, unit))
