"""Data loader for ZeroFlow predictions with Waymo tfrecord ground truth.

Reads pre-computed ZeroFlow scene flow predictions from ``.feather`` files,
point clouds from OpenPCDet-style ``.pkl`` files, and bounding boxes /
timestamps from the original Waymo ``.tfrecord`` files.  Yields
``(SceneFlowPair, pred_flow, is_valid)`` tuples aligned row-for-row.

The ``.pkl`` point clouds are ground-removed and subsampled (~30K pts/frame)
compared to the full tfrecord point clouds (~124K pts/frame).  The loader
builds ``SceneFlowPair`` instances using pkl point clouds so that the
predicted flow vectors align 1:1 with ``pair.points_t0``.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from rigid_flow.core.types import SceneFlowPair

logger = logging.getLogger(__name__)

SEGMENT_PREFIX = "segment-"
SEGMENT_SUFFIX = "_with_camera_labels"


def _strip_segment_name(dirname: str) -> str:
    """Convert directory name to the tfrecord ``context.name`` format.

    ``segment-XXXX_with_camera_labels`` -> ``XXXX``
    """
    name = dirname
    if name.startswith(SEGMENT_PREFIX):
        name = name[len(SEGMENT_PREFIX):]
    if name.endswith(SEGMENT_SUFFIX):
        name = name[: -len(SEGMENT_SUFFIX)]
    return name


class ZeroFlowDataSource:
    """Iterates frame pairs from pkl + feather files enriched with tfrecord boxes.

    Args:
        pkl_root: Root directory containing per-segment subdirectories of
            ``{frame_idx:06d}.pkl`` files (e.g. ``validation/segment-XXX/``).
        feather_root: Root directory containing per-segment subdirectories of
            ``{pair_idx:010d}.feather`` files (e.g. ``sequence_len_002/segment-XXX/``).
        tfrecord_root: Root directory searched recursively for ``.tfrecord``
            files.  Only segments that also appear in *pkl_root* are loaded.
    """

    def __init__(
        self,
        pkl_root: Path,
        feather_root: Path,
        tfrecord_root: Path,
    ) -> None:
        self.pkl_root = Path(pkl_root)
        self.feather_root = Path(feather_root)
        self.tfrecord_root = Path(tfrecord_root)

        self._segment_dirs = self._discover_segments()
        if not self._segment_dirs:
            raise FileNotFoundError(
                f"No matching segments found across pkl={self.pkl_root}, "
                f"feather={self.feather_root}, tfrecord={self.tfrecord_root}"
            )
        logger.info("ZeroFlowDataSource: %d segments discovered", len(self._segment_dirs))

    def _discover_segments(
        self,
    ) -> list[dict]:
        """Match segments across the three data sources.

        Returns a list of dicts with keys ``segment_name``, ``context_name``,
        ``pkl_dir``, ``feather_dir``, ``tfrecord_path``.
        """
        pkl_dirs = {
            d.name: d for d in sorted(self.pkl_root.iterdir()) if d.is_dir()
        }
        feather_dirs = {
            d.name: d for d in sorted(self.feather_root.iterdir()) if d.is_dir()
        }

        tfrecord_paths = sorted(self.tfrecord_root.rglob("*.tfrecord"))
        # Index tfrecords by both the raw stem and the stripped context name
        # so we can match regardless of whether the file retains the
        # ``segment-`` prefix and ``_with_camera_labels`` suffix.
        tfrecord_by_key: dict[str, Path] = {}
        for p in tfrecord_paths:
            tfrecord_by_key[p.stem] = p
            tfrecord_by_key[_strip_segment_name(p.stem)] = p

        matched: list[dict] = []
        for seg_name in sorted(pkl_dirs):
            if seg_name not in feather_dirs:
                logger.warning("Segment %s in pkl but missing from feather; skipping", seg_name)
                continue
            context_name = _strip_segment_name(seg_name)
            # Try full directory name first, then stripped context name.
            tfrecord_path = tfrecord_by_key.get(seg_name) or tfrecord_by_key.get(context_name)
            if tfrecord_path is None:
                logger.warning(
                    "Segment %s (context=%s) has no tfrecord; skipping",
                    seg_name,
                    context_name,
                )
                continue
            matched.append(
                {
                    "segment_name": seg_name,
                    "context_name": context_name,
                    "pkl_dir": pkl_dirs[seg_name],
                    "feather_dir": feather_dirs[seg_name],
                    "tfrecord_path": tfrecord_path,
                }
            )
        return matched

    def iterate_pairs(
        self,
    ) -> Iterator[tuple[SceneFlowPair, NDArray[np.float32], NDArray[np.bool_]]]:
        """Yield ``(pair, pred_flow, is_valid)`` for every consecutive frame pair.

        ``pair.points_t0`` comes from the pkl ``car_frame_pc``; boxes and
        timestamps come from the tfrecord.  ``pred_flow`` is the ZeroFlow
        prediction from the feather file, aligned row-for-row with
        ``pair.points_t0``.
        """
        import tensorflow as tf
        from rigid_flow.data.waymo_protos import dataset_pb2

        for seg_info in self._segment_dirs:
            yield from self._iterate_segment(seg_info, tf, dataset_pb2)

    def _iterate_segment(
        self,
        seg_info: dict,
        tf,
        dataset_pb2,
    ) -> Iterator[tuple[SceneFlowPair, NDArray[np.float32], NDArray[np.bool_]]]:
        seg_name = seg_info["segment_name"]
        context_name = seg_info["context_name"]
        pkl_dir: Path = seg_info["pkl_dir"]
        feather_dir: Path = seg_info["feather_dir"]
        tfrecord_path: Path = seg_info["tfrecord_path"]

        logger.info("Loading segment %s", seg_name)

        pkl_files = sorted(pkl_dir.glob("*.pkl"))
        feather_files = sorted(feather_dir.glob("*.feather"))

        num_pairs = len(feather_files)
        if len(pkl_files) < num_pairs + 1:
            logger.warning(
                "Segment %s: expected >= %d pkl files for %d feather pairs, got %d; skipping",
                seg_name, num_pairs + 1, num_pairs, len(pkl_files),
            )
            return

        tfrecord_frames = self._parse_tfrecord_frames(tfrecord_path, tf, dataset_pb2)
        if len(tfrecord_frames) < num_pairs + 1:
            logger.warning(
                "Segment %s: tfrecord has %d frames but need %d; skipping",
                seg_name, len(tfrecord_frames), num_pairs + 1,
            )
            return

        prev_pkl = self._load_pkl(pkl_files[0])

        for pair_idx in range(num_pairs):
            cur_pkl = self._load_pkl(pkl_files[pair_idx + 1])
            feather_data = self._load_feather(feather_files[pair_idx])

            tf_t0 = tfrecord_frames[pair_idx]
            tf_t1 = tfrecord_frames[pair_idx + 1]

            n_pkl = prev_pkl["car_frame_pc"].shape[0]
            n_feather = feather_data["pred_flow"].shape[0]
            if n_pkl != n_feather:
                raise ValueError(
                    f"Segment {seg_name} pair {pair_idx}: pkl has {n_pkl} points "
                    f"but feather has {n_feather} rows"
                )

            pair = SceneFlowPair(
                points_t0=prev_pkl["car_frame_pc"],
                points_t1=cur_pkl["car_frame_pc"],
                ego_pose_t0=prev_pkl["pose"],
                ego_pose_t1=cur_pkl["pose"],
                boxes_t0=tf_t0["boxes"],
                boxes_t1=tf_t1["boxes"],
                timestamp_us_t0=tf_t0["timestamp_us"],
                timestamp_us_t1=tf_t1["timestamp_us"],
                gt_flow=None,
                sequence_id=context_name,
                frame_index=pair_idx,
            )

            yield pair, feather_data["pred_flow"], feather_data["is_valid"]

            prev_pkl = cur_pkl

    @staticmethod
    def _parse_tfrecord_frames(
        tfrecord_path: Path,
        tf,
        dataset_pb2,
    ) -> list[dict]:
        """Parse all frames from a tfrecord, extracting only boxes and timestamps."""
        from rigid_flow.data.waymo_parser import WaymoParser

        dummy_parser = WaymoParser.__new__(WaymoParser)
        dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")

        frames: list[dict] = []
        for raw_record in dataset:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytes(raw_record.numpy()))
            boxes = dummy_parser._extract_boxes(frame)
            timestamp_us = frame.timestamp_micros
            frames.append({"boxes": boxes, "timestamp_us": timestamp_us})

        return frames

    @staticmethod
    def _load_pkl(path: Path) -> dict:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return {
            "car_frame_pc": data["car_frame_pc"].astype(np.float32),
            "pose": data["pose"].astype(np.float64),
        }

    @staticmethod
    def _load_feather(path: Path) -> dict:
        df = pd.read_feather(path)
        pred_flow = df[["flow_tx_m", "flow_ty_m", "flow_tz_m"]].values.astype(np.float32)
        is_valid = df["is_valid"].values.astype(np.bool_)
        return {"pred_flow": pred_flow, "is_valid": is_valid}
