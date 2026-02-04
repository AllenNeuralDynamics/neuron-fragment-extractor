"""
Created on Mon Feb 2 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

TO DO

"""

from segmentation_skeleton_metrics.evaluate import evaluate
from segmentation_skeleton_metrics.utils.img_util import TensorStoreReader

import pandas as pd
import os
import shutil
import zipfile

from segmentation_skeleton_metrics.utils import util


def main():
    # Extract fragments
    evaluate_segmentation()

    # Load merge sites
    merge_sites_path = os.path.join(input_dir, "results-merge_sites.csv")
    all_merge_sites = pd.read_csv(merge_sites_path)

    # Build dataset directory
    print("\nExtracting merge site dataset...")
    fragments_dir = os.path.join(input_dir, "fragments")
    for gt_filename in util.list_dir(fragments_dir, ".zip"):
        # Create dataset directory
        print("Filename:", gt_filename)
        gt_name, _ = os.path.splitext(gt_filename)
        neuron_output_dir = os.path.join(output_dir, gt_name)
        util.mkdir(neuron_output_dir)

        # Copy files
        copy_gt_swc(gt_name, neuron_output_dir)
        copy_fragments(gt_name, neuron_output_dir)

        # Copy fragment subsets
        idx_mask = all_merge_sites["GroundTruth_ID"] == gt_name
        merge_sites = all_merge_sites[idx_mask]
        merged_ids = get_merge_ids(merge_sites)

        copy_fragments_subset(gt_name, neuron_output_dir, merged_ids, True)
        copy_fragments_subset(gt_name, neuron_output_dir, merged_ids, False)

        # Copy merge sites
        gt_id = get_gt_id(gt_name)
        merge_names = get_merge_names(merge_sites)
        if len(merge_names) > 0:
            copy_merge_sites(neuron_output_dir, merge_names, gt_id)

        path = os.path.join(neuron_output_dir, f"merge_sites_{brain_id}-{gt_id}.csv")
        merge_sites.to_csv(path, index=False)

    # Clean up dataset directory
    path = os.path.join(output_dir, "merge_sites.csv")
    all_merge_sites.to_csv(path, index=False)
    shutil.rmtree(input_dir)


def copy_gt_swc(gt_name, dst_dir):
    src_zip = os.path.join(input_dir, "fragments", f"{gt_name}.zip")
    with zipfile.ZipFile(src_zip, "r") as zf:
        src_name = f"{gt_name}.swc"
        dst_path = os.path.join(dst_dir, f"{gt_name}.swc")
        with zf.open(src_name) as src, open(dst_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def copy_fragments(gt_name, dst_dir):
    # Paths
    gt_id = get_gt_id(gt_name)
    src_zip = os.path.join(input_dir, "fragments", f"{gt_name}.zip")
    dst_zip = os.path.join(dst_dir, f"fragments-{brain_id}.{gt_id}.zip")

    # Parse files
    exclude_name = f"{gt_name}.swc"
    with zipfile.ZipFile(src_zip, "r") as zin, \
         zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == exclude_name:
                continue
            zout.writestr(item, zin.read(item.filename))


def copy_fragments_subset(gt_name, dataset_dir, merged_ids, is_merges):
    # Parameters
    gt_id = get_gt_id(gt_name)
    is_contained = is_merged_fragment if is_merges else is_nonmerged_fragment
    name = "fragments_merged" if is_merges else "fragments_nonmerged"

    # Paths
    src_zip = os.path.join(input_dir, "fragments", f"{gt_name}.zip")
    dst_zip = os.path.join(dataset_dir, f"{name}-{brain_id}.{gt_id}.zip")
    exclude_name = f"{gt_name}.swc"

    # Parse files
    with zipfile.ZipFile(src_zip, "r") as zin, \
         zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            # Check if GT SWC file
            if item.filename == exclude_name:
                continue

            # Check whether to write
            swc_name, _ = os.path.splitext(item.filename)
            if is_contained(merged_ids, swc_name):
                zout.writestr(item, zin.read(item.filename))


def copy_merge_sites(dataset_dir, merge_names, gt_id):
    # Paths
    src_zip = os.path.join(input_dir, "results-merged_fragments.zip")
    dst_zip = os.path.join(dataset_dir, f"merge_sites-{brain_id}.{gt_id}.zip")

    # Parse files
    with zipfile.ZipFile(src_zip, "r") as zin, \
         zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            # Check whether to write
            if item.filename in merge_names:
                zout.writestr(item, zin.read(item.filename))


def is_merged_fragment(merged_ids, swc_name):
    return swc_name in merged_ids


def is_nonmerged_fragment(merged_ids, swc_name):
    return swc_name not in merged_ids


# --- Helpers ---
def evaluate_segmentation():
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


def get_merge_ids(merge_sites):
    return set([str(sid) for sid in merge_sites["Segment_ID"]])


def get_merge_names(merge_sites):
    return set([str(sid) for sid in merge_sites["Merge_ID"]])


def print_experiment_details():
    # Initialize log path
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
    input_dir = f"/home/jupyter/results/merge_datasets/{brain_id}/temp"
    output_dir = f"/home/jupyter/results/merge_datasets/{brain_id}/{segmentation_id}"
    util.mkdir(output_dir, delete=True)

    # Run
    main()
