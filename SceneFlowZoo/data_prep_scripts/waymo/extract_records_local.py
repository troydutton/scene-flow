"""
Simplified extraction script for processing individual Waymo tfrecord files
into the .pkl format expected by bucketed-scene-flow-eval / SceneFlowZoo.

Unlike extract_flow_and_remove_ground.py, this script:
  - Takes a flat directory of .tfrecord files (no training/validation split needed)
  - Takes a heightmap directory organized as <heightmap_root>/<segment_name>_map/
  - Outputs to <save_directory>/<segment_name>/<idx:06d>.pkl

Usage:
    python extract_records_local.py \
        --tfrecord_dir /data/troy/datasets/raw/waymo \
        --heightmap_dir /data/troy/datasets/waymo_open_processed_flow/heightmaps/validation \
        --save_dir /data/troy/datasets/waymo_open_processed_flow/validation \
        --cpus 8

Dependencies (install in a TF env):
    pip install tensorflow waymo-open-dataset-tf-2-12-0 open3d numpy joblib \
                bucketed-scene-flow-eval
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import argparse
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from typing import Tuple

from bucketed_scene_flow_eval.datastructures import PointCloud, SE3, SE2
from bucketed_scene_flow_eval.utils import load_json, save_pickle

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils

GROUND_HEIGHT_THRESHOLD = 0.4  # 40 cm


# ---------------------------------------------------------------------------
# Heightmap helpers
# ---------------------------------------------------------------------------

def load_ground_height_raster(map_path: Path):
    raster_height_path = map_path / "ground_height.npy"
    transform_path = map_path / "se2.json"
    raster_heightmap = np.load(raster_height_path)
    transform = load_json(transform_path)
    transform_rotation = np.array(transform["R"]).reshape(2, 2)
    transform_translation = np.array(transform["t"])
    transform_scale = np.array(transform["s"])
    transform_se2 = SE2(rotation=transform_rotation, translation=transform_translation)
    return raster_heightmap, transform_se2, transform_scale


def is_ground_points(raster_heightmap, global_to_raster_se2, global_to_raster_scale, global_point_cloud: PointCloud) -> np.ndarray:
    global_points_xy = global_point_cloud.points[:, :2]
    raster_points_xy = (
        global_to_raster_se2.transform_point_cloud(global_points_xy) * global_to_raster_scale
    )
    raster_points_xy = np.round(raster_points_xy).astype(np.int64)

    ground_height_values = np.full((raster_points_xy.shape[0],), np.nan)
    outside = (
        (raster_points_xy[:, 0] >= raster_heightmap.shape[1])
        | (raster_points_xy[:, 1] >= raster_heightmap.shape[0])
        | (raster_points_xy[:, 0] < 0)
        | (raster_points_xy[:, 1] < 0)
    )
    ind_valid = ~outside
    ground_height_values[ind_valid] = raster_heightmap[
        raster_points_xy[ind_valid, 1], raster_points_xy[ind_valid, 0]
    ]

    is_ground = (
        np.abs(global_point_cloud.points[:, 2] - ground_height_values) <= GROUND_HEIGHT_THRESHOLD
    ) | ((global_point_cloud.points[:, 2] - ground_height_values) < 0)
    return is_ground


# ---------------------------------------------------------------------------
# Waymo tfrecord parsing
# ---------------------------------------------------------------------------

def parse_range_image_and_camera_projection(frame):
    """Parse range images, camera projections, and scene flow from a frame."""
    range_images = {}
    camera_projections = {}
    point_flows = {}
    range_image_top_pose = None

    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:
            ri_str = tf.io.decode_compressed(laser.ri_return1.range_image_compressed, "ZLIB")
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytes(ri_str.numpy()))
            range_images[laser.name] = [ri]

            if len(laser.ri_return1.range_image_flow_compressed) > 0:
                flow_str = tf.io.decode_compressed(laser.ri_return1.range_image_flow_compressed, "ZLIB")
                ri_flow = dataset_pb2.MatrixFloat()
                ri_flow.ParseFromString(bytes(flow_str.numpy()))
                point_flows[laser.name] = [ri_flow]

            if laser.name == dataset_pb2.LaserName.TOP:
                pose_str = tf.io.decode_compressed(laser.ri_return1.range_image_pose_compressed, "ZLIB")
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(bytes(pose_str.numpy()))

            cp_str = tf.io.decode_compressed(laser.ri_return1.camera_projection_compressed, "ZLIB")
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytes(cp_str.numpy()))
            camera_projections[laser.name] = [cp]

        if len(laser.ri_return2.range_image_compressed) > 0:
            ri_str = tf.io.decode_compressed(laser.ri_return2.range_image_compressed, "ZLIB")
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytes(ri_str.numpy()))
            range_images[laser.name].append(ri)

            if len(laser.ri_return2.range_image_flow_compressed) > 0:
                flow_str = tf.io.decode_compressed(laser.ri_return2.range_image_flow_compressed, "ZLIB")
                ri_flow = dataset_pb2.MatrixFloat()
                ri_flow.ParseFromString(bytes(flow_str.numpy()))
                point_flows[laser.name].append(ri_flow)

            cp_str = tf.io.decode_compressed(laser.ri_return2.camera_projection_compressed, "ZLIB")
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytes(cp_str.numpy()))
            camera_projections[laser.name].append(cp)

    return range_images, camera_projections, point_flows, range_image_top_pose


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, point_flows, range_image_top_pose, ri_index=0, keep_polar_features=True):
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    flows = []

    cartesian_range_images = frame_utils.convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, keep_polar_features=True
    )

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = tf.gather_nd(range_image_cartesian, tf.compat.v1.where(range_image_mask))

        n_points = int(points_tensor.shape[0])

        # Flow is optional — default to zeros (4 cols: vx, vy, vz, class) if not present
        if c.name in point_flows and ri_index < len(point_flows[c.name]):
            flow = point_flows[c.name][ri_index]
            flow_tensor = tf.reshape(tf.convert_to_tensor(value=flow.data), flow.shape.dims)
            flow_points_tensor = tf.gather_nd(flow_tensor, tf.compat.v1.where(range_image_mask))
        else:
            flow_points_tensor = tf.zeros((n_points, 4), dtype=tf.float32)

        points.append(points_tensor.numpy())
        flows.append(flow_points_tensor.numpy())

    return points, flows


def get_car_pc_global_pc_flow_transform(frame):
    range_images, camera_projections, point_flows, range_image_top_poses = (
        parse_range_image_and_camera_projection(frame)
    )

    points_lst, flows_lst = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, point_flows, range_image_top_poses,
        keep_polar_features=True
    )

    car_frame_pc = points_lst[0][:, 3:]   # (N, 3) XYZ in vehicle frame
    car_frame_flows = flows_lst[0][:, :3]  # (N, 3) vx, vy, vz (zeros if no GT flow)
    car_frame_labels = flows_lst[0][:, 3]  # (N,) semantic class (zeros if no GT flow)

    num_points = car_frame_pc.shape[0]
    world_frame_pc = np.concatenate([car_frame_pc, np.ones([num_points, 1])], axis=-1)
    car_to_global_transform = np.reshape(np.array(frame.pose.transform), [4, 4])
    world_frame_pc = np.transpose(
        np.matmul(car_to_global_transform, np.transpose(world_frame_pc))
    )[:, 0:3]

    offset = frame.map_pose_offset
    points_offset = np.array([offset.x, offset.y, offset.z])
    world_frame_pc += points_offset

    return (
        PointCloud(car_frame_pc),
        PointCloud(world_frame_pc),
        car_frame_flows,
        car_frame_labels,
        SE3.from_array(car_to_global_transform),
    )


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record(file_path: Path, heightmap_dir: Path, save_dir: Path):
    file_path = Path(file_path)
    seg_name = file_path.stem.replace(".tfrecord", "")
    if seg_name.endswith("_with_camera_labels"):
        seg_name = seg_name  # keep full name as-is

    heightmap_path = heightmap_dir / f"{seg_name}_map"
    save_folder = save_dir / seg_name

    if not heightmap_path.exists():
        print(f"[SKIP] No heightmap found for {seg_name} at {heightmap_path}")
        return

    save_folder.mkdir(parents=True, exist_ok=True)

    # Skip if already fully processed (check if non-empty)
    existing = list(save_folder.glob("*.pkl"))
    if len(existing) > 0:
        print(f"[SKIP] {seg_name} already has {len(existing)} pkl files, skipping")
        return

    print(f"[START] Processing {seg_name}")
    raster_heightmap, transform_se2, transform_scale = load_ground_height_raster(heightmap_path)

    dataset = tf.data.TFRecordDataset(str(file_path), compression_type="")
    saved_count = 0
    for idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame.FromString(bytes(data.numpy()))

        try:
            car_frame_pc, global_frame_pc, flow, label, pose = get_car_pc_global_pc_flow_transform(frame)
        except Exception as e:
            print(f"  [WARN] Frame {idx} failed: {type(e).__name__}: {e}")
            continue

        keep_points_mask = ~is_ground_points(
            raster_heightmap, transform_se2, transform_scale, global_frame_pc
        )

        masked_car_frame_pc = car_frame_pc.mask_points(keep_points_mask)
        # Flow is in dm/frame in Waymo (when GT is present); normalize to m/frame.
        # When flow is zero (no GT), this is a no-op.
        masked_flow = flow[keep_points_mask] / 10.0
        masked_label = label[keep_points_mask]

        save_pickle(
            save_folder / f"{idx:06d}.pkl",
            {
                "car_frame_pc": masked_car_frame_pc.points,
                "flow": masked_flow,
                "label": masked_label,
                "pose": pose.to_array(),
                "fraction_kept": float(np.sum(keep_points_mask)) / len(keep_points_mask),
            },
        )
        saved_count += 1

    print(f"[DONE] {seg_name}: saved {saved_count} frames")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_dir", type=Path, required=True,
                        help="Directory containing *.tfrecord files (flat, no split subfolders)")
    parser.add_argument("--heightmap_dir", type=Path, required=True,
                        help="Directory containing <segment_name>_map/ heightmap folders")
    parser.add_argument("--save_dir", type=Path, required=True,
                        help="Output directory for .pkl files")
    parser.add_argument("--cpus", type=int, default=1,
                        help="Number of parallel workers (default 1; TF can be finicky with multiprocessing)")
    parser.add_argument("--segment", type=str, default=None,
                        help="Optional: process only this segment name (without .tfrecord)")
    args = parser.parse_args()

    assert args.tfrecord_dir.is_dir(), f"tfrecord_dir {args.tfrecord_dir} does not exist"
    assert args.heightmap_dir.is_dir(), f"heightmap_dir {args.heightmap_dir} does not exist"
    args.save_dir.mkdir(parents=True, exist_ok=True)

    all_records = sorted(args.tfrecord_dir.glob("*.tfrecord"))
    assert len(all_records) > 0, f"No .tfrecord files found in {args.tfrecord_dir}"

    if args.segment:
        all_records = [r for r in all_records if args.segment in r.stem]
        assert len(all_records) > 0, f"No records matching segment '{args.segment}'"

    print(f"Processing {len(all_records)} records with {args.cpus} workers")

    if args.cpus == 1:
        for record in all_records:
            process_record(record, args.heightmap_dir, args.save_dir)
    else:
        Parallel(n_jobs=args.cpus)(
            delayed(process_record)(r, args.heightmap_dir, args.save_dir)
            for r in all_records
        )


if __name__ == "__main__":
    main()
