"""Infer missing tooth sites from Stage 1 detections.

DENTEX (and Stage 1 in general) only annotates/detects teeth that are physically
present. Missing teeth are inferred by diffing the expected adult FDI set against
whatever Stage 1 actually detected. Wisdom teeth (18/28/38/48) are excluded from
the expected set since their absence is common and not itself abnormal.
"""

from __future__ import annotations

from .fdi_mapping import FDI_TOOTH_NUMBERS

WISDOM_TEETH = {"18", "28", "38", "48"}
EXPECTED_FDI_SET: set[str] = set(FDI_TOOTH_NUMBERS) - WISDOM_TEETH


def find_missing_teeth(detected_fdi_numbers: list[str]) -> list[str]:
    """Return the sorted list of expected FDI numbers absent from detections.

    Each entry is a candidate missing-tooth site requiring implant assessment.
    """
    detected = set(detected_fdi_numbers)
    missing = EXPECTED_FDI_SET - detected
    return sorted(missing, key=lambda fdi: (fdi[0], fdi[1]))
