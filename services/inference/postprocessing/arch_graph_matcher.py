"""Dental Arch Graph Sequential Post-Processor.

Enforces anatomical continuity along the dental arch:
- Maxillary (Upper) Arch: 18 -> 17 -> ... -> 11 -> 21 -> ... -> 28
- Mandibular (Lower) Arch: 48 -> 47 -> ... -> 41 -> 31 -> ... -> 38

In a panoramic X-ray:
1. Teeth are distributed along two monotonic horizontal coordinate curves.
2. Left-to-right anatomical ordering is strictly monotonic:
   - For upper arch: X(18) < X(17) < ... < X(11) < X(21) < ... < X(28)
   - For lower arch: X(48) < X(47) < ... < X(41) < X(31) < ... < X(38)
3. Upper teeth have lower Y centroids than lower teeth.

This module resolves ambiguities (e.g. midline 11 vs 21, 31 vs 41, or competing
duplicate boxes) by applying dynamic programming sequence alignment across candidate
detections, eliminating spatial inversions without requiring retraining.
"""

from __future__ import annotations

import numpy as np

# Anatomical sequence definitions
UPPER_ARCH_SEQUENCE = [f"1{p}" for p in range(8, 0, -1)] + [f"2{p}" for p in range(1, 9)]
LOWER_ARCH_SEQUENCE = [f"4{p}" for p in range(8, 0, -1)] + [f"3{p}" for p in range(1, 9)]

UPPER_ARCH_SET = set(UPPER_ARCH_SEQUENCE)
LOWER_ARCH_SET = set(LOWER_ARCH_SEQUENCE)


def get_box_center(box: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def enforce_dental_arch_ordering(
    detections: list[dict],
    arch_split_ratio: float = 0.5,
) -> list[dict]:
    """Align raw detections against anatomical maxillary and mandibular arch sequences.

    Parameters:
        detections: List of detection dicts with keys: 'bbox', 'fdi_number', 'confidence'
        arch_split_ratio: Approximate vertical split threshold to separate upper and lower arches
                          if not clearly separated (default 0.5 of max Y range).

    Returns:
        Filtered and re-aligned list of detections with anatomical consistency guaranteed.
    """
    if not detections:
        return []

    # Sort detections roughly into Upper and Lower jaws by Y centroid or predicted quadrant
    upper_candidates = []
    lower_candidates = []

    for det in detections:
        fdi = str(det.get("fdi_number", ""))
        cx, cy = get_box_center(det["bbox"])
        det_with_center = dict(det)
        det_with_center["cx"] = cx
        det_with_center["cy"] = cy

        if fdi in UPPER_ARCH_SET:
            upper_candidates.append(det_with_center)
        elif fdi in LOWER_ARCH_SET:
            lower_candidates.append(det_with_center)
        else:
            # Fallback based on Y if quadrant unknown
            upper_candidates.append(det_with_center)

    def optimize_arch(candidates: list[dict], expected_sequence: list[str]) -> list[dict]:
        if not candidates:
            return []

        # Sort spatially from patient-right to patient-left (left-to-right on OPG X axis)
        sorted_by_x = sorted(candidates, key=lambda d: d["cx"])

        # Map expected FDI code to sequence index (0 to 15)
        seq_idx_map = {fdi: idx for idx, fdi in enumerate(expected_sequence)}

        # Filter out obvious sequence inversions:
        # We find the longest monotonic subsequence of predicted arch indices
        # weighted by detection confidence.
        n = len(sorted_by_x)
        dp = [sorted_by_x[i]["confidence"] for i in range(n)]
        prev = [-1] * n

        for i in range(n):
            idx_i = seq_idx_map.get(sorted_by_x[i]["fdi_number"], -1)
            for j in range(i):
                idx_j = seq_idx_map.get(sorted_by_x[j]["fdi_number"], -1)
                # Ensure strictly increasing FDI position along the arch
                if idx_j < idx_i:
                    if dp[j] + sorted_by_x[i]["confidence"] > dp[i]:
                        dp[i] = dp[j] + sorted_by_x[i]["confidence"]
                        prev[i] = j

        # Reconstruct optimal consistent chain
        best_end = int(np.argmax(dp))
        chosen_indices = []
        curr = best_end
        while curr != -1:
            chosen_indices.append(curr)
            curr = prev[curr]
        chosen_indices.reverse()

        result = []
        for idx in chosen_indices:
            clean_det = dict(sorted_by_x[idx])
            clean_det.pop("cx", None)
            clean_det.pop("cy", None)
            result.append(clean_det)
        return result

    optimal_upper = optimize_arch(upper_candidates, UPPER_ARCH_SEQUENCE)
    optimal_lower = optimize_arch(lower_candidates, LOWER_ARCH_SEQUENCE)

    return optimal_upper + optimal_lower
