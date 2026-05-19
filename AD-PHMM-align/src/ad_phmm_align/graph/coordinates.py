"""Coordinate, packed-window, and edge-overlap contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ad_phmm_align.exceptions import ArtifactValidationError


@dataclass(frozen=True)
class PackedStateWindows:
    """Packed half-open PHMM state windows for graph nodes.

    `left` and `right` are global half-open intervals `[left, right)`.
    `offset` points into packed per-node buffers and `length` must equal
    `right - left`.
    """

    left: np.ndarray
    right: np.ndarray
    offset: np.ndarray
    length: np.ndarray

    def validate(self, global_state_count: int) -> None:
        """Validate half-open windows and packed offsets."""

        validate_state_intervals(self.left, self.right, global_state_count)
        if not (
            self.left.shape
            == self.right.shape
            == self.offset.shape
            == self.length.shape
        ):
            raise ArtifactValidationError("packed window arrays must match")
        if np.any(self.offset < 0):
            raise ArtifactValidationError("packed window offset is negative")
        if self.offset.size and int(self.offset[0]) != 0:
            raise ArtifactValidationError("packed window offsets must start at 0")
        if self.offset.size > 1:
            expected_offset = self.offset[:-1] + self.length[:-1]
            if np.any(self.offset[1:] != expected_offset):
                raise ArtifactValidationError(
                    "packed window offsets must be contiguous with preceding lengths"
                )
        expected = self.right - self.left
        if np.any(self.length != expected):
            raise ArtifactValidationError(
                "packed window length must equal right - left"
            )


@dataclass(frozen=True)
class EdgeWindowOverlaps:
    """Packed overlap metadata for source/target node state windows."""

    edge_ids: np.ndarray
    src_offset: np.ndarray
    dst_offset: np.ndarray
    length: np.ndarray

    def validate(self, edge_count: int) -> None:
        """Validate edge-window overlap array shapes and bounds."""

        if not (
            self.edge_ids.shape
            == self.src_offset.shape
            == self.dst_offset.shape
            == self.length.shape
        ):
            raise ArtifactValidationError("edge overlap arrays must match")
        if np.any(self.edge_ids < 0):
            raise ArtifactValidationError("edge overlap id is negative")
        if self.edge_ids.size and np.max(self.edge_ids) >= edge_count:
            raise ArtifactValidationError("edge overlap id exceeds edge_count")
        if np.any(self.src_offset < 0) or np.any(self.dst_offset < 0):
            raise ArtifactValidationError("edge overlap offset is negative")
        if np.any(self.length < 0):
            raise ArtifactValidationError("edge overlap length is negative")

    def validate_against_windows(
        self,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        windows: PackedStateWindows,
    ) -> None:
        """Validate overlap offsets against source/destination node windows."""

        self.validate(int(edge_src.shape[0]))
        if edge_src.shape != edge_dst.shape:
            raise ArtifactValidationError("edge source/destination arrays must match")
        for edge_id, src_offset, dst_offset, length in zip(
            self.edge_ids.tolist(),
            self.src_offset.tolist(),
            self.dst_offset.tolist(),
            self.length.tolist(),
        ):
            src_node = int(edge_src[int(edge_id)])
            dst_node = int(edge_dst[int(edge_id)])
            if src_node >= windows.length.shape[0] or dst_node >= windows.length.shape[0]:
                raise ArtifactValidationError(
                    "edge overlap references a node without state-window metadata"
                )
            if int(src_offset) + int(length) > int(windows.length[src_node]):
                raise ArtifactValidationError("edge overlap exceeds source state window")
            if int(dst_offset) + int(length) > int(windows.length[dst_node]):
                raise ArtifactValidationError("edge overlap exceeds destination state window")


def validate_state_intervals(
    left: np.ndarray, right: np.ndarray, global_state_count: int
) -> None:
    """Validate half-open `[left, right)` intervals against global states."""

    if left.shape != right.shape:
        raise ArtifactValidationError("state interval arrays must have same shape")
    if np.any(left < 0):
        raise ArtifactValidationError("state interval left bound is negative")
    if np.any(right < left):
        raise ArtifactValidationError("state interval right bound is before left")
    if np.any(left > global_state_count):
        raise ArtifactValidationError("state interval left bound exceeds global state count")
    if np.any(right > global_state_count):
        raise ArtifactValidationError("state interval exceeds global state count")
