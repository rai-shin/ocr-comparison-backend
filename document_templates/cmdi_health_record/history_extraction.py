import cv2
import numpy as np

from document_templates.cmdi_health_record.history_tables import (
    MEDICAL_HISTORY,
    FAMILY_HISTORY,
)


TABLE_LAYOUTS = {
    "medical_history": {
        "definition": MEDICAL_HISTORY,
        "x": [21.60, 129.38, 156.38, 178.94],
        "y": [
            226.82, 235.34, 243.86, 252.38, 261.02,
            269.54, 278.06, 286.58, 295.25, 303.77,
            312.29, 320.81, 329.45, 337.97, 346.49,
            355.01, 363.65, 372.17, 380.69, 389.21,
            397.85, 406.37, 414.89, 423.41, 432.05,
            440.57, 449.09, 457.61, 466.13, 474.77,
            483.31, 491.83, 500.35, 508.99, 517.51,
        ],
    },
    "family_history": {
        "definition": FAMILY_HISTORY,
        "x": [280.61, 406.63, 442.63, 478.66],
        "y": [
            225.98, 234.50, 243.02, 251.66,
            260.18, 268.70, 277.22, 285.86,
            294.41, 302.93, 311.45, 320.09,
            328.61, 337.13,
        ],
    },
}


def detect_cell_mark(image, x0, x1, y0, y1):
    """
    Detect a preliminary handwritten slash mark
    inside one YES or NO cell.

    Detection thresholds scale with image resolution.
    """

    scale_x = image.shape[1] / 612
    scale_y = image.shape[0] / 936

    # Reference resolution: 1700 x 2600.
    reference_scale_x = 1700 / 612
    reference_scale_y = 2600 / 936

    relative_x = scale_x / reference_scale_x
    relative_y = scale_y / reference_scale_y

    margin_x = max(1, round(7 * relative_x))
    margin_y = max(1, round(7 * relative_y))

    left = round(x0 * scale_x) + margin_x
    right = round(x1 * scale_x) - margin_x
    top = round(y0 * scale_y) + margin_y
    bottom = round(y1 * scale_y) - margin_y

    crop = image[top:bottom, left:right]

    if crop.size == 0:
        return False

    _, binary = cv2.threshold(
        crop,
        170,
        255,
        cv2.THRESH_BINARY_INV,
    )

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(binary, 8)
    )

    min_area = max(1, round(12 * relative_x * relative_y))
    min_width = max(1, round(3 * relative_x))
    min_height = max(1, round(7 * relative_y))

    for component in range(1, count):
        bx, by, width, height, area = stats[component]

        if (
            area >= min_area
            and height >= min_height
            and width >= min_width
            and height / max(width, 1) >= 1.5
        ):
            return True

    return False


def classify_response(no_mark, yes_mark):
    if no_mark and yes_mark:
        return "NEEDS_REVIEW"

    if no_mark:
        return "NO"

    if yes_mark:
        return "YES"

    return "UNMARKED"


def extract_history_tables(image):
    """
    Extract preliminary YES/NO responses from the
    CMDI Health Record Medical and Family History tables.

    Images are normalized to the reference resolution
    before mark detection. The original input is not modified.
    """

    # Validate that an image was supplied.
    if image is None:
        raise ValueError("An image is required.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("The supplied image is empty.")

    if image.ndim not in (2, 3):
        raise ValueError(
            "Unsupported image dimensions. "
            "Expected a grayscale or color image."
        )

    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(
            "Image width and height must be greater than zero."
        )

    # Convert supported color formats to grayscale.
    if image.ndim == 3:
        channels = image.shape[2]

        if channels == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )
        elif channels == 4:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY,
            )
        else:
            raise ValueError(
                "Unsupported number of image channels. "
                "Expected 3 (BGR) or 4 (BGRA)."
            )

    # Check whether the original image meets the recommended resolution.
    minimum_width = 1530
    minimum_height = 2340

    reference_width = 1700
    reference_height = 2600

    if (
        image.shape[1] < minimum_width
        or image.shape[0] < minimum_height
    ):
        raise ValueError(
            "Image resolution is below the recommended minimum "
            f"of {minimum_width}x{minimum_height}. "
            f"Received {image.shape[1]}x{image.shape[0]}. "
            "Use a higher-resolution image for reliable mark detection."
        )

    # Normalize to the reference resolution.
    if (
        image.shape[1] != reference_width
        or image.shape[0] != reference_height
    ):
        image = cv2.resize(
            image,
            (reference_width, reference_height),
            interpolation=cv2.INTER_AREA,
        )

    extracted = {}

    for table_name, layout in TABLE_LAYOUTS.items():
        definition = layout["definition"]
        x = layout["x"]
        y = layout["y"]

        rows = []

        for index, label in enumerate(definition["rows"]):
            no_mark = detect_cell_mark(
                image,
                x[1],
                x[2],
                y[index + 1],
                y[index + 2],
            )

            yes_mark = detect_cell_mark(
                image,
                x[2],
                x[3],
                y[index + 1],
                y[index + 2],
            )

            response = classify_response(
                no_mark,
                yes_mark,
            )

            rows.append({
                "label": label,
                "response": response,
                "no_mark_detected": no_mark,
                "yes_mark_detected": yes_mark,
            })

        extracted[table_name] = rows

    return extracted
