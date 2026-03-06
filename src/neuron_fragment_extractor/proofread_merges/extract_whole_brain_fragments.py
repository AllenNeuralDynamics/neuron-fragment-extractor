"""
Created on Mon Feb 2 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that extracts fragments intersecting with ground truth neuron tracings
along with their detected merge sites. The following outputs are extracted:
    /{brain_id}
        /{segmentation_id}
            /{neuron_id}
                * {neuron_id}.swc
                * fragments-{neuron_id}.zip
                * merge_sites-{neuron_id}.csv
                * merge_sites-{neuron_id}.zip
"""

from segmentation_skeleton_metrics.evaluate import evaluate
from segmentation_skeleton_metrics.utils.img_util import TensorStoreReader

import pandas as pd
import os
import shutil

from neuron_fragment_extractor.utils import util


def main():
    # Extract fragments and merge sites
    #extract_via_skeleton_metrics()
    merge_sites = pd.read_csv(f"{input_dir}/results-merge_sites.csv")

    # Build dataset
    print("\nBuild Dataset...")
    fragments_dir = os.path.join(input_dir, "fragments")
    for gt_filename in util.list_dir(fragments_dir, ".zip"):
        # Neuron info
        print("Filename:", gt_filename)
        gt_name, _ = os.path.splitext(gt_filename)
        neuron_output_dir = os.path.join(output_dir, get_gt_id(gt_name))
        util.mkdir(neuron_output_dir)

        # Copy SWC files
        copy_gt_swc(gt_name, neuron_output_dir)
        copy_fragment_swcs(gt_name, neuron_output_dir)
        copy_merged_fragment_swcs(gt_name, neuron_output_dir, merge_sites)
        copy_merge_site_swcs(gt_name, neuron_output_dir, merge_sites)
        copy_nonmerged_fragment_swcs(gt_name, neuron_output_dir, merge_sites)

        # Write merge site CSV
        idxs = merge_sites["GroundTruth_ID"] == gt_name
        path = f"{neuron_output_dir}/merge_sites-{get_gt_id(gt_name)}.csv"
        merge_sites[idxs].to_csv(path, index=False)

    # Clean up dataset directory
    path = os.path.join(output_dir, "merge_sites.csv")
    merge_sites.to_csv(path, index=False)
    shutil.rmtree(input_dir)


def copy_gt_swc(gt_name, dst_dir):
    # Set paths
    src_zip = f"{input_dir}/fragments/{gt_name}.zip"
    src_name = f"{gt_name}.swc"
    dst_path = f"{dst_dir}/{get_gt_id(gt_name)}.swc"

    # Move files
    util.copy_file_from_zip(src_zip, src_name, dst_path)


def copy_fragment_swcs(gt_name, dst_dir):
    # Set paths
    src_zip = f"{input_dir}/fragments/{gt_name}.zip"
    src_names = util.list_zip_filenames(src_zip)
    src_names.remove(f"{gt_name}.swc")
    dst_zip = f"{dst_dir}/fragments-{get_gt_id(gt_name)}.zip"

    # Move files
    util.copy_files_from_zip(src_zip, src_names, dst_zip)


def copy_merged_fragment_swcs(gt_name, dst_dir, merge_sites):
    # Set paths
    src_zip = f"{input_dir}/fragments/{gt_name}.zip"
    src_names = get_merged_names(gt_name, merge_sites)
    dst_zip = f"{dst_dir}/fragments_merged-{get_gt_id(gt_name)}.zip"

    # Move files
    util.copy_files_from_zip(src_zip, src_names, dst_zip)


def copy_nonmerged_fragment_swcs(gt_name, dst_dir, merge_sites):
    # Set paths
    src_zip = f"{input_dir}/fragments/{gt_name}.zip"
    src_names = get_nonmerged_names(gt_name, merge_sites)
    dst_zip = f"{dst_dir}/fragments_nonmerged-{get_gt_id(gt_name)}.zip"

    # Move files
    util.copy_files_from_zip(src_zip, src_names, dst_zip)


def copy_merge_site_swcs(gt_name, dst_dir, merge_sites):
    # Set paths
    src_zip = f"{input_dir}/results-merged_fragments.zip"
    src_names = get_site_names(gt_name, merge_sites)
    dst_zip = f"{dst_dir}/merge_sites-{get_gt_id(gt_name)}.zip"

    # Move files
    util.copy_files_from_zip(src_zip, src_names, dst_zip)


# --- Helpers ---
def extract_via_skeleton_metrics():
    # Paths
    fragments_path = f"gs://{bucket_name}/from_google/{brain_id}/whole_brain/{segmentation_id}/swcs"
    gt_path = f"gs://{bucket_name}/ground_truth_tracings/{brain_id}/voxel/"
    segmentation_path = f"gs://{bucket_name}/from_google/{brain_id}/whole_brain/{segmentation_id}/"

    # Load data
    segmentation = TensorStoreReader(segmentation_path)

    # Run evaluation
    print_experiment_details()
    evaluate(
        gt_path,
        segmentation,
        input_dir,
        anisotropy=anisotropy,
        fragments_path=fragments_path,
        results_filename=results_filename,
        save_fragments=save_fragments,
        save_merges=save_merges,
    )


def get_gt_id(gt_name):
    return gt_name.split("-")[0]


def get_merged_names(gt_name, merge_sites):
    idxs = merge_sites["GroundTruth_ID"] == gt_name
    return set([str(sid) for sid in merge_sites[idxs]["Segment_ID"]])


def get_nonmerged_names(gt_name, merge_sites):
    merged_names = get_merged_names(gt_name, merge_sites)
    names = set([str(sid) for sid in merge_sites["Segment_ID"]])
    return names - merged_names


def get_site_names(gt_name, merge_sites):
    idxs = merge_sites["GroundTruth_ID"] == gt_name
    return set([str(mid) for mid in merge_sites[idxs]["Merge_ID"]])


def print_experiment_details():
    # Log path
    log_path = os.path.join(output_dir, f"{results_filename}-overview.txt")
    if os.path.exists(log_path):
        os.remove(log_path)

    # Report experiment details
    util.update_txt(log_path, "\nExperiment Details")
    util.update_txt(log_path, "-" * (len(segmentation_id) + 9))
    util.update_txt(log_path, f"Brain_ID: {brain_id}")
    util.update_txt(log_path, f"Segmentation_ID: {segmentation_id}")
    util.update_txt(log_path, f"Evaluate Corrected: {evaluate_corrected}")


if __name__ == "__main__":
    # Dataset
    brain_id = "802449"
    segmentation_id = "jin_masked_mean40_stddev105"

    # Parameters
    anisotropy = (0.748, 0.748, 1.0)
    bucket_name = "allen-nd-goog"
    evaluate_corrected = False
    save_fragments = True
    results_filename = "results"
    save_merges = True

    # Paths
    input_dir = f"/home/jupyter/results/merge_datasets/{brain_id}/test"
    output_dir = f"/home/jupyter/results/merge_datasets/{brain_id}/{segmentation_id}"
    util.mkdir(output_dir, delete=True)

    # Run
    main()
