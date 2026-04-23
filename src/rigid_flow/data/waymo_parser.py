"""Waymo Open Dataset parser for rigid scene flow correction.

Parses Waymo tfrecord files and yields consecutive frame pairs as
SceneFlowPair instances for downstream scene flow estimation.

Uses compiled protobuf definitions directly (no waymo-open-dataset SDK
required), making this compatible with Apple Silicon / ARM64.
"""

from __future__ import annotations

import logging
import tarfile
import zlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rigid_flow.core.types import BoundingBox, SceneFlowPair

logger = logging.getLogger(__name__)


class WaymoParser:
    """Parses Waymo Open Dataset tfrecord files into SceneFlowPair objects."""

    def __init__(self, data_root: Path) -> None:
        """Find all .tfrecord files under *data_root*.

        Parameters
        ----------
        data_root:
            Directory (possibly nested) containing ``.tfrecord`` files.
        """
        self.data_root = Path(data_root)
        self.tfrecord_paths: list[Path] = sorted(self.data_root.rglob("*.tfrecord"))
        if not self.tfrecord_paths:
            raise FileNotFoundError(
                f"No .tfrecord files found under {self.data_root}"
            )
        logger.info("Found %d tfrecord files in %s", len(self.tfrecord_paths), self.data_root)

    # ------------------------------------------------------------------
    # Static helper – extract tar archive
    # ------------------------------------------------------------------

    @staticmethod
    def extract_tar(tar_path: Path, dest: Path) -> Path:
        """Extract a tar archive to *dest* and return the extracted directory.

        Parameters
        ----------
        tar_path:
            Path to ``.tar`` (or ``.tar.gz``) archive containing tfrecord files.
        dest:
            Destination directory for extraction.

        Returns
        -------
        Path
            Directory inside *dest* that contains the tfrecord files.  If the
            archive has a single top-level directory, that directory is returned;
            otherwise *dest* itself is returned.
        """
        tar_path = Path(tar_path)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting %s -> %s", tar_path, dest)
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(dest)

        # Determine extracted root: if there is a single top-level directory,
        # return it; otherwise return dest.
        with tarfile.open(tar_path, "r:*") as tar:
            top_level = {Path(m.name).parts[0] for m in tar.getmembers() if Path(m.name).parts}
        if len(top_level) == 1:
            candidate = dest / next(iter(top_level))
            if candidate.is_dir():
                return candidate
        return dest

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def iterate_pairs(self) -> Iterator[SceneFlowPair]:
        """Yield consecutive frame pairs from every tfrecord sequence.

        Each tfrecord file is treated as one driving sequence.  Frames are
        streamed (never fully materialised in memory) and a sliding window of
        size 2 produces the pairs.
        """
        import tensorflow as tf
        from rigid_flow.data.waymo_protos import dataset_pb2

        for tfrecord_path in self.tfrecord_paths:
            logger.info("Processing %s", tfrecord_path.name)
            dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")

            prev_frame_data: tuple[NDArray, NDArray, list[BoundingBox], int] | None = None
            frame_idx = 0
            sequence_id = ""

            for raw_record in dataset:
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytes(raw_record.numpy()))

                if not sequence_id:
                    sequence_id = frame.context.name

                cur_frame_data = self._parse_frame(frame)

                if prev_frame_data is not None:
                    points_t0, ego_pose_t0, boxes_t0, ts_t0 = prev_frame_data
                    points_t1, ego_pose_t1, boxes_t1, ts_t1 = cur_frame_data

                    yield SceneFlowPair(
                        points_t0=points_t0,
                        points_t1=points_t1,
                        ego_pose_t0=ego_pose_t0,
                        ego_pose_t1=ego_pose_t1,
                        boxes_t0=boxes_t0,
                        boxes_t1=boxes_t1,
                        timestamp_us_t0=ts_t0,
                        timestamp_us_t1=ts_t1,
                        gt_flow=None,
                        sequence_id=sequence_id,
                        frame_index=frame_idx,
                    )
                    frame_idx += 1

                prev_frame_data = cur_frame_data

    # ------------------------------------------------------------------
    # Per-frame parsing
    # ------------------------------------------------------------------

    def _parse_frame(
        self, frame
    ) -> tuple[NDArray[np.float32], NDArray[np.float64], list[BoundingBox], int]:
        """Extract point cloud, ego pose, bounding boxes, and timestamp.

        Parameters
        ----------
        frame:
            A ``dataset_pb2.Frame`` proto.

        Returns
        -------
        points : (N, 3) float32
        ego_pose : (4, 4) float64
        boxes : list of BoundingBox
        timestamp_us : int
        """
        points = self._range_images_to_points(frame)
        ego_pose = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
        boxes = self._extract_boxes(frame)
        timestamp_us = frame.timestamp_micros
        return points, ego_pose, boxes, timestamp_us

    def _range_images_to_points(self, frame) -> NDArray[np.float32]:
        """Convert range images from all LiDARs to a merged (N, 3) point cloud.

        Performs spherical-to-Cartesian conversion using beam inclinations and
        sensor extrinsics from the frame calibration.  Only the first return is
        used.  Invalid returns (range <= 0) are discarded.
        """
        from rigid_flow.data.waymo_protos import dataset_pb2

        calibrations = {cal.name: cal for cal in frame.context.laser_calibrations}
        all_points: list[NDArray[np.float32]] = []

        for laser in frame.lasers:
            compressed = laser.ri_return1.range_image_compressed
            if len(compressed) == 0:
                continue

            cal = calibrations[laser.name]

            # Decompress and parse the MatrixFloat protobuf.
            raw = zlib.decompress(compressed)
            ri_proto = dataset_pb2.MatrixFloat()
            ri_proto.ParseFromString(raw)

            H, W, C = list(ri_proto.shape.dims)
            ri = np.array(ri_proto.data, dtype=np.float32).reshape(H, W, C)

            ranges = ri[:, :, 0]  # channel 0 = range

            # Beam inclinations (elevation angles per row).
            if len(cal.beam_inclinations) > 0:
                inclinations = np.array(cal.beam_inclinations)
            else:
                inclinations = np.linspace(
                    cal.beam_inclination_min, cal.beam_inclination_max, H
                )

            # Azimuth angles (uniform from pi to -pi across columns).
            azimuths = np.linspace(np.pi, -np.pi, W, endpoint=False)

            incl_grid, az_grid = np.meshgrid(inclinations, azimuths, indexing="ij")

            # Spherical → Cartesian in sensor frame.
            cos_incl = np.cos(incl_grid)
            x = ranges * cos_incl * np.cos(az_grid)
            y = ranges * cos_incl * np.sin(az_grid)
            z = ranges * np.sin(incl_grid)

            valid = ranges > 0
            pts_sensor = np.stack([x[valid], y[valid], z[valid]], axis=-1)

            # Sensor frame → vehicle (ego) frame.
            extrinsic = np.array(cal.extrinsic.transform).reshape(4, 4)
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            pts_vehicle = (R @ pts_sensor.T).T + t

            all_points.append(pts_vehicle.astype(np.float32))

        if not all_points:
            return np.empty((0, 3), dtype=np.float32)

        return np.concatenate(all_points, axis=0)

    def _extract_boxes(self, frame) -> list[BoundingBox]:
        """Convert ``frame.laser_labels`` to a list of :class:`BoundingBox`.

        Waymo label types:
            1 = TYPE_VEHICLE
            2 = TYPE_PEDESTRIAN
            3 = TYPE_SIGN
            4 = TYPE_CYCLIST
        """
        boxes: list[BoundingBox] = []
        for label in frame.laser_labels:
            box = label.box
            center = np.array(
                [box.center_x, box.center_y, box.center_z], dtype=np.float32
            )
            dimensions = np.array(
                [box.length, box.width, box.height], dtype=np.float32
            )
            heading = float(box.heading)
            class_label = int(label.type)
            tracking_id = str(label.id)

            # Velocity – always populated (zero for stationary).
            meta = label.metadata
            velocity = np.array([meta.speed_x, meta.speed_y], dtype=np.float32)

            boxes.append(
                BoundingBox(
                    center=center,
                    dimensions=dimensions,
                    heading=heading,
                    class_label=class_label,
                    tracking_id=tracking_id,
                    velocity=velocity,
                )
            )
        return boxes
