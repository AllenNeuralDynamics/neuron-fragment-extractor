"""
Created on Wed June 5 16:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Given a GCS directory containing ground truth segmentation blocks, this code
extracts SWC files of segments from predicted segmentations that are contained
in these blocks.

"""

from neuron_proofreader.utils.img_util import TensorStoreReader

import fastremap
import gcsfs
import os

from neuron_fragment_extractor.utils import swc_util, util


def main():
    # Initializations
    segmentation_dirs = util.list_gcs_subdirectories(
        bucket_name, "from_google/whole_brain/"
    )

    # Iterate over groundtruth blocks
    gt_subdirs = util.list_gcs_subdirectories(bucket_name, gt_blocks_prefix)
    for gt_subdir in gt_subdirs:
        # Extract metadata
        metadata_path = os.path.join(gt_subdir, "metadata.json")
        metadata = util.read_json_from_gcs(bucket_name, metadata_path)
        brain_id = get_brain_id(metadata)
        block_id = gt_subdir.split("/")[-2]
        if int(block_id[-2]) != 0:
            continue
        print("Dataset:", brain_id, block_id)

        # Store ground truth tracings
        input_dir = gt_subdir.replace("blocks", "swcs")
        output_dir = get_output_dir(brain_id, block_id, None, True)
        store_groundtruth_swcs(input_dir, output_dir, metadata)
        continue

        # Extract fragments by segmentation
        swc_reader = swc_util.Reader(anisotropy, min_size)
        for segmentation_dir in find_matching_dirs(segmentation_dirs, brain_id):
            # Initialize output directory
            segmentation_id = segmentation_dir.split("/")[-2]
            output_dir = get_output_dir(brain_id, block_id, segmentation_id)
            print("Output Directory:", output_dir)

            # Extract labels from label_mask
            segment_ids = get_segment_ids(segmentation_dir, metadata)
            print("# Labels Found:", len(segment_ids))

            # Search fragments for matching labels
            swc_dicts = get_swc_dicts(swc_reader, brain_id, segmentation_dir)
            while swc_dicts:
                swc_dict = swc_dicts.pop()
                segment_id = swc_util.get_segment_id(swc_dict["swc_name"])
                if segment_id in segment_ids:
                    path = os.path.join(output_dir, swc_dict["swc_name"] + ".swc")
                    swc_util.write_swc(swc_dict, path)
            print("# Fragments Found:", util.count_files(output_dir))
            print("")


def store_groundtruth_swcs(input_dir, output_dir, metadata):
    for swc_path in util.list_gcs_filenames(bucket_name, input_dir, ".swc"):
        input_path = os.path.join(f"gs://{bucket_name}", swc_path)
        output_path = os.path.join(output_dir, os.path.basename(swc_path))
        download_swc(input_path, output_path, metadata)


# --- Helpers ---
def find_matching_dirs(segmentation_dirs, target_brain_id):
    # Find matching directory
    matching_dir = None
    for segmentation_dir in segmentation_dirs:
        brain_id = segmentation_dir.split("/")[-2]
        if brain_id == target_brain_id:
            matching_dir = segmentation_dir
            break

    # Get matching segmentation directories with label mask
    if matching_dir:
        dirs_with_label_mask = list()
        for d in util.list_gcs_subdirectories(bucket_name, matching_dir):
            if util.gcs_directory_exists(bucket_name, d):
                dirs_with_label_mask.append(d)
        return dirs_with_label_mask
    else:
        return list()


def find_swc_dirname(segmentation_dir):
    for subdir in util.list_gcs_subdirectories(bucket_name, segmentation_dir):
        dirname = subdir.split("/")[4]
        if "swc" in dirname:
            return dirname
    return None


def get_brain_id(metadata):
    img_dirname = metadata["image_url"].split("/")[3]
    return img_dirname.split("_")[1]


def get_segment_ids(segmentation_dir, metadata):
    # Initializations
    label_mask_path = os.path.join("gs://allen-nd-goog", segmentation_dir, "label_mask/")
    label_mask_reader = TensorStoreReader(label_mask_path)
    origin = metadata["chunk_origin"][::-1]
    shape = metadata["chunk_shape"][::-1]

    # Read labels
    center = tuple([o + s // 2 for o, s in zip(origin, shape)])
    patch = label_mask_reader.read(center, shape)
    return set(fastremap.unique(patch).astype(int))


def get_swc_dicts(swc_reader, brain_id, segmentation_dir):
    swc_dirname = find_swc_dirname(segmentation_dir)
    swc_dir = os.path.join(segmentation_dir, swc_dirname)
    fragments_pointer = f"gs://{bucket_name}/{swc_dir}"
    return swc_reader.read(fragments_pointer)


def get_output_dir(brain_id, block_id, segmentation_id, is_gt=False):
    # Set directory path
    if is_gt:
        path = f"{output_dir}/{brain_id}/target_swcs/{block_id}/final_trees"
    else:
        path = f"{output_dir}/{brain_id}/pred_swcs/{segmentation_id}/{block_id}"

    # Create directory
    os.makedirs(path, exist_ok=True)
    return path


def download_swc(source_path, dst_path, metadata):
    fs = gcsfs.GCSFileSystem()
    with fs.open(source_path, 'r') as f:
        converted_lines = list()
        ox, oy, oz = metadata["chunk_origin"][::-1]
        sx, sy, sz = anisotropy
        for line in f:
            if line.startswith("#"):
                # Try to parse offset from comment
                tokens = line.strip("#").strip().split()
                if len(tokens) >= 4 and tokens[0].upper() == "OFFSET":
                    oxx, oyy, ozz = map(float, tokens[1:4])
                    ox += oxx
                    oy += oyy
                    oz += ozz
                else:
                    converted_lines.append(line)
                continue

            parts = line.strip().split()
            if len(parts) != 7:
                continue  # skip malformed lines

            nid, ntype = parts[0], parts[1]
            x, y, z = map(float, parts[2:5])
            radius = parts[5]
            parent = parts[6]

            # Apply offset, then convert local voxel → physical
            vz = (x + ox) * sz
            vy = (y + oy) * sy
            vx = (z + oz) * sx

            converted_lines.append(
                f"{nid} {ntype} {vx:.6f} {vy:.6f} {vz:.6f} {radius} {parent}\n"
            )

    with open(dst_path, "w") as f:
        f.writelines(converted_lines)


if __name__ == "__main__":
    # Parameters
    anisotropy = (0.748, 0.748, 1.0)
    min_size = 35

    # Paths
    bucket_name = "allen-nd-goog"
    gt_blocks_prefix = "from_aind/training-data_2025-06-10/blocks/"
    output_dir = "/home/jupyter/data/split_proofreading"

    # Run
    main()
