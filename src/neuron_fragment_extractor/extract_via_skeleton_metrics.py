"""
Created on Mon Feb 2 12:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code that extracts fragments that intersect with ground truth tracings by
using the GitHub repository:
    https://github.com/AllenNeuralDynamics/segmentation-skeleton-metrics

"""

from segmentation_skeleton_metrics.evaluate import evaluate
from segmentation_skeleton_metrics.utils.img_util import TensorStoreReader

import argparse


def main():
    """
    Extracts fragments that intersect with ground truth tracings and saves the
    results in "output_dir".
    """
    # Paths
    fragments_path = f"gs://allen-nd-goog/from_google/{brain_id}/whole_brain/{segmentation_id}/swcs"
    gt_path = f"gs://allen-nd-goog/ground_truth_tracings/{brain_id}/voxel/"
    segmentation_path = f"gs://allen-nd-goog/from_google/{brain_id}/whole_brain/{segmentation_id}/"

    # Load data
    segmentation = TensorStoreReader(segmentation_path)

    # Run evaluation
    evaluate(
        gt_path,
        segmentation,
        output_dir,
        anisotropy=(0.748, 0.748, 1.0),
        fragments_path=fragments_path,
        save_fragments=True,
        save_merges=True
    )


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--brain_id")
    parser.add_argument("--segmentation_id")
    parser.add_argument("--output_dir")

    # Run code
    args = parser.parse_args()
    brain_id = args.brain_id
    segmentation_id = args.segmentation_id
    output_dir = args.output_dir

    main()
