"""Predicted-bounding-box source for the rigid-flow pipeline.

Parses a Waymo Open Dataset detection-submission ``.bin`` file (a serialized
``waymo_open_dataset.protos.metrics_pb2.Objects`` message, or a concatenation
of such messages) into a per-frame index of
:class:`rigid_flow.core.types.BoundingBox` objects keyed by
``(context_name, frame_timestamp_micros)``.

Notes on the Waymo detection bin format
---------------------------------------
The canonical form is a single ``Objects`` message whose ``objects`` field is
a ``repeated Object``.  Because protobuf caps a single message at 2 GiB, large
submissions are sometimes stored as a **concatenation** of ``Objects``
messages.  For the ``repeated`` field this is semantically identical because
``ParseFromString`` on the concatenated bytes simply appends each message's
``objects`` into a single growing list.  We therefore parse the whole file
with one ``ParseFromString`` call; if that fails we fall back to a
length-delimited reader for defensiveness.

Known limitations
-----------------
* CenterPoint detection submissions typically leave
  ``object.metadata.speed_x/speed_y`` at 0.  In that case the
  ``epe_static/slow/fast`` breakdown in :mod:`rigid_flow.eval.metrics` will
  push nearly all predicted-box foreground mass into the ``static`` bucket.
  This is a property of the upstream detector, not of this module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
from google.protobuf.internal.decoder import _DecodeVarint
from google.protobuf.message import DecodeError

from rigid_flow.core.types import BoundingBox
from rigid_flow.data.waymo_protos.protos import metrics_pb2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level framing
# ---------------------------------------------------------------------------


def _iter_objects(blob: bytes) -> Iterator[metrics_pb2.Object]:
    """Yield every ``Object`` in *blob*, tolerating multi-message concatenation.

    Strategy:
        1. Try a single ``Objects.ParseFromString(blob)`` — handles both the
           canonical single-message case and simple concatenations (because
           ``repeated`` fields merge).
        2. If that raises :class:`DecodeError`, fall back to a length-delimited
           reader (each record = varint length + serialized ``Objects``).
    """
    objs = metrics_pb2.Objects()
    try:
        objs.ParseFromString(blob)
        logger.info(
            "Parsed detection bin as a single Objects message: %d Object entries",
            len(objs.objects),
        )
        yield from objs.objects
        return
    except DecodeError as exc:
        logger.warning("Single-message parse failed (%s); retrying length-delimited", exc)

    pos = 0
    total = len(blob)
    record_idx = 0
    while pos < total:
        length, new_pos = _DecodeVarint(blob, pos)
        pos = new_pos
        end = pos + length
        if end > total:
            raise ValueError(
                f"Length-delimited record {record_idx} extends past end of file "
                f"(pos={pos}, length={length}, total={total})"
            )
        record = metrics_pb2.Objects()
        record.ParseFromString(blob[pos:end])
        yield from record.objects
        pos = end
        record_idx += 1
    logger.info("Parsed detection bin as %d length-delimited Objects records", record_idx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PredBoxIndex:
    """In-memory index of predicted boxes keyed by ``(context_name, timestamp_us)``.

    Each stored value is a list of ``(BoundingBox, score)`` tuples sorted by
    descending score, so a score-threshold filter is a prefix scan.
    """

    def __init__(self, bin_path: Path) -> None:
        bin_path = Path(bin_path)
        logger.info("Loading predicted boxes from %s (%.1f MiB)", bin_path, bin_path.stat().st_size / (1 << 20))
        with open(bin_path, "rb") as f:
            blob = f.read()

        index: dict[tuple[str, int], list[tuple[BoundingBox, float]]] = {}
        for i, obj in enumerate(_iter_objects(blob)):
            box = obj.object.box
            meta = obj.object.metadata
            velocity = np.array([meta.speed_x, meta.speed_y], dtype=np.float32)
            if float(velocity[0]) == 0.0 and float(velocity[1]) == 0.0:
                velocity_out: np.ndarray | None = None
            else:
                velocity_out = velocity

            bounding_box = BoundingBox(
                center=np.array([box.center_x, box.center_y, box.center_z], dtype=np.float32),
                dimensions=np.array([box.length, box.width, box.height], dtype=np.float32),
                heading=float(box.heading),
                class_label=int(obj.object.type),
                tracking_id=f"pred_{obj.context_name}_{obj.frame_timestamp_micros}_{i}",
                velocity=velocity_out,
            )
            key = (str(obj.context_name), int(obj.frame_timestamp_micros))
            index.setdefault(key, []).append((bounding_box, float(obj.score)))

        # Sort each frame's boxes by descending score so thresholding is a prefix scan.
        for entries in index.values():
            entries.sort(key=lambda bs: bs[1], reverse=True)

        self._index = index
        self._bin_path = bin_path
        self._num_boxes = sum(len(v) for v in index.values())

        logger.info(
            "PredBoxIndex: %d frames, %d total predicted boxes",
            len(index),
            self._num_boxes,
        )
        sample_keys = list(index.keys())[:3]
        for k in sample_keys:
            logger.info("  sample key: context=%s ts=%d -> %d boxes", k[0], k[1], len(index[k]))

    # ------------------------------------------------------------------

    @property
    def num_frames(self) -> int:
        return len(self._index)

    @property
    def num_boxes(self) -> int:
        return self._num_boxes

    def frame_keys(self) -> Iterable[tuple[str, int]]:
        return self._index.keys()

    def get(
        self,
        sequence_id: str,
        timestamp_us: int,
        score_threshold: float = 0.0,
    ) -> list[BoundingBox]:
        """Return predicted boxes for one frame, filtered by *score_threshold*.

        Parameters
        ----------
        sequence_id:
            Waymo ``context_name`` (matches ``SceneFlowPair.sequence_id`` from
            :class:`rigid_flow.data.waymo_parser.WaymoParser`).
        timestamp_us:
            Frame timestamp in microseconds (matches
            ``SceneFlowPair.timestamp_us_t0``).
        score_threshold:
            Predictions with ``score < threshold`` are dropped.

        Returns
        -------
        list[BoundingBox]
            Possibly empty.  Because entries are score-sorted, the returned
            list is also score-sorted (highest first).
        """
        entries = self._index.get((str(sequence_id), int(timestamp_us)))
        if not entries:
            return []
        return [bb for bb, score in entries if score >= score_threshold]
