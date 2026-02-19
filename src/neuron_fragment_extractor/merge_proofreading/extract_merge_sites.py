"""
Created on Mon Feb 2 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that extracts merge sites from fragments that intersect with ground truth
neuron tracings. The following are extracted:
    /{output_dir}
        * gt_neurons.zip
        * fragments.zip
        * fragments_merged.zip
        * fragments_nonmerged.zip
        * merge_sites.csv
        * merge_sites.zip
"""

import argparse
import pandas as pd
import os

from neuron_fragment_extractor.utils import util


def main():
    # Initializations
    fragments_dir = os.path.join(input_dir, "fragments")
    merge_sites = pd.read_csv(f"{input_dir}/results-merge_sites.csv")

    # Copy SWC files
    for gt_filename in util.list_dir(fragments_dir, ".zip"):
        # Neuron info
        gt_name, _ = os.path.splitext(gt_filename)
        copy_gt_swc(gt_name)
        copy_fragment_swcs(gt_name)

    # Copy subsets of SWC files
    copy_merged_fragment_swcs(merge_sites)
    copy_nonmerged_fragment_swcs(merge_sites)
    copy_merge_site_swcs(merge_sites)

    # Copy merge sites CSV
    path = os.path.join(output_dir, "merge_sites.csv")
    merge_sites.to_csv(path, index=False)
    print("# Merge Sites:", len(merge_sites))


def copy_gt_swc(gt_name):
    # Set paths
    src_zip = f"{input_dir}/fragments/{gt_name}.zip"
    src_names = {f"{gt_name}.swc"}
    dst_zip = f"{output_dir}/gt_neurons.zip"

    # Move files
    util.copy_files_from_zip(src_zip, src_names, dst_zip)


def copy_fragment_swcs(gt_name):
    # Set paths
    src_zip = f"{input_dir}/fragments/{gt_name}.zip"
    src_names = util.list_zip_filenames(src_zip)
    src_names.remove(f"{gt_name}.swc")
    dst_zip = f"{output_dir}/fragments.zip"

    # Move files
    util.copy_files_from_zip(src_zip, src_names, dst_zip)


def copy_merged_fragment_swcs(merge_sites):
    # Set paths
    src_zip = f"{output_dir}/fragments.zip"
    src_names = get_merged_names(merge_sites)
    dst_zip = f"{output_dir}/fragments_merged.zip"

    # Move files
    util.copy_files_from_zip(src_zip, src_names, dst_zip)


def copy_nonmerged_fragment_swcs(merge_sites):
    # Set paths
    src_zip = f"{output_dir}/fragments.zip"
    src_names = get_nonmerged_names(merge_sites)
    dst_zip = f"{output_dir}/fragments_nonmerged.zip"

    # Move files
    util.copy_files_from_zip(src_zip, src_names, dst_zip)


def copy_merge_site_swcs(merge_sites):
    # Set paths
    src_zip = f"{input_dir}/results-merged_fragments.zip"
    src_names = get_site_names(merge_sites)
    dst_zip = f"{output_dir}/merge_sites.zip"

    # Move files
    util.copy_files_from_zip(src_zip, src_names, dst_zip)


# --- Helpers ---
def get_gt_id(gt_name):
    return gt_name.split("-")[0]


def get_merged_names(merge_sites):
    return set([f"{sid}.swc" for sid in merge_sites["Segment_ID"]])


def get_nonmerged_names(merge_sites):
    merged_names = get_merged_names(merge_sites)
    names = util.list_zip_filenames(f"{output_dir}/fragments.zip")
    return set(names) - merged_names


def get_site_names(merge_sites):
    return set([str(mid) for mid in merge_sites["Merge_ID"]])


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")

    # Run code
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    main()
