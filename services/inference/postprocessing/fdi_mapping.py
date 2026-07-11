"""FDI two-digit tooth numbering scheme (ISO 3950).

Quadrant 1: upper right (11-18), Quadrant 2: upper left (21-28),
Quadrant 3: lower left (31-38), Quadrant 4: lower right (41-48).
Deciduous teeth use quadrants 5-8 and are out of scope for the initial adult model.
"""

FDI_TOOTH_NUMBERS: list[str] = [
    f"{quadrant}{position}"
    for quadrant in (1, 2, 3, 4)
    for position in range(1, 9)
]

FDI_CLASS_TO_INDEX = {fdi: i for i, fdi in enumerate(FDI_TOOTH_NUMBERS)}
INDEX_TO_FDI_CLASS = {i: fdi for fdi, i in FDI_CLASS_TO_INDEX.items()}


def index_to_fdi(class_index: int) -> str:
    if class_index not in INDEX_TO_FDI_CLASS:
        raise ValueError(f"Unknown class index: {class_index}")
    return INDEX_TO_FDI_CLASS[class_index]
