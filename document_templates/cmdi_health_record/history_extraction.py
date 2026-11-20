
import cv2
import numpy as np
import json
from pathlib import Path

from document_templates.cmdi_health_record.history_tables import (
    MEDICAL_HISTORY,
    FAMILY_HISTORY,
    STUDENT_EXPERIENCE,
    CIRCUMCISION,
    PSYCHOSOCIAL_HISTORY,
)


BASE_DIR = Path(__file__).resolve().parents[2]
FIELDS_PATH = BASE_DIR / "health_record_fields.json"

REFERENCE_WIDTH = 1700
REFERENCE_HEIGHT = 2600


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


FIELD_MAPPINGS = {
    "student_experiences": [
        ("student_experiences", "hospitalization", "Hospitalization", "no", "yes"),
        ("student_experiences", "surgery_operation", "Surgery / Operation", "no", "yes"),
        ("student_experiences", "special_medication", "On special medication", "no", "yes"),
    ],
    "circumcision": [
        ("for_boys", "circumcision", "Circumcision", "done", "not_done"),
    ],
    "psychosocial_history": [
        ("personal_social_history", "close_friend", "Do you have a close friend?", "yes", "no"),
        ("personal_social_history", "drive", "Do you drive?", "yes", "no"),
        ("personal_social_history", "drive_regularly", "Do you drive regularly?", "yes", "no"),
        ("personal_social_history", "drink_alcohol", "Do you drink alcohol bev.?", "yes", "no"),
        ("personal_social_history", "smoke", "Do you smoke?", "yes", "no"),
        ("personal_social_history", "illicit_drugs", "Have you taken illicit drugs?", "yes", "no"),
        ("personal_social_history", "regular_drug_use", "Regular Drug Use", "yes", "no"),
        ("personal_social_history", "experienced_abuse", "Experienced abuse?", "yes", "no"),
        ("abuse_details", "physical_abuse", "Physical", "yes", "no"),
        ("abuse_details", "sexual_abuse", "Sexual", "yes", "no"),
        ("abuse_details", "verbal_abuse", "Verbal", "yes", "no"),
    ],
}


GIRLS_TEXT_FIELDS = [
    ("age_of_menarche", "Age of Menarche", "text"),
    ("last_menstrual_period", "Last Menstrual Period", "date"),
    ("regular_cycle_days", "Regular Cycle Days", "text"),
    ("irregular_cycle_days", "Irregular Cycle Days", "text"),
]


GIRLS_CHOICE_FIELDS = {
    "menstrual_cycle": {
        "label": "Cycle",
        "options": {
            "option_1": "REGULAR",
            "option_2": "IRREGULAR",
        },
    },
    "menstrual_flow": {
        "label": "Flow",
        "options": {
            "option_1": "MINIMAL",
            "option_2": "MODERATE",
            "option_3": "PROFUSE",
        },
    },
}


def load_health_record_fields():
    if not FIELDS_PATH.exists():
        raise FileNotFoundError(
            f"Health record field mappings not found: {FIELDS_PATH}"
        )

    with open(FIELDS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def find_field(data, section_id, field_id):
    for section in data.get("sections", []):
        if section.get("id") == section_id:
            for field in section.get("fields", []):
                if field.get("id") == field_id:
                    return field

    raise ValueError(
        f"Field mapping not found: {section_id}.{field_id}"
    )


def detect_region_mark(image, region):
    """Detect a handwritten mark inside a mapped checkbox region."""

    scale_x = image.shape[1] / 612
    scale_y = image.shape[0] / 936

    reference_scale_x = REFERENCE_WIDTH / 612
    reference_scale_y = REFERENCE_HEIGHT / 936

    relative_x = scale_x / reference_scale_x
    relative_y = scale_y / reference_scale_y

    margin_x = max(1, round(2 * relative_x))
    margin_y = max(1, round(2 * relative_y))

    left = round(region["x0"] * scale_x) + margin_x
    right = round(region["x1"] * scale_x) - margin_x
    top = round(region["y0"] * scale_y) + margin_y
    bottom = round(region["y1"] * scale_y) - margin_y

    left = max(0, left)
    top = max(0, top)
    right = min(image.shape[1], right)
    bottom = min(image.shape[0], bottom)

    crop = image[top:bottom, left:right]

    if crop.size == 0:
        return False

    _, binary = cv2.threshold(
        crop, 170, 255, cv2.THRESH_BINARY_INV
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


def detect_cell_mark(image, x0, x1, y0, y1):
    """Detect a mark using the original A/B table coordinates."""

    scale_x = image.shape[1] / 612
    scale_y = image.shape[0] / 936

    reference_scale_x = REFERENCE_WIDTH / 612
    reference_scale_y = REFERENCE_HEIGHT / 936

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
        crop, 170, 255, cv2.THRESH_BINARY_INV
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


def classify_response(first_mark, second_mark, first_value, second_value):
    if first_mark and second_mark:
        return "NEEDS_REVIEW"

    if first_mark:
        return first_value

    if second_mark:
        return second_value

    return "UNMARKED"


def extract_original_tables(image, extracted):
    for table_name, layout in TABLE_LAYOUTS.items():
        definition = layout["definition"]
        x = layout["x"]
        y = layout["y"]

        rows = []

        for index, label in enumerate(definition["rows"]):
            no_mark = detect_cell_mark(
                image,
                x[1], x[2],
                y[index + 1], y[index + 2],
            )

            yes_mark = detect_cell_mark(
                image,
                x[2], x[3],
                y[index + 1], y[index + 2],
            )

            response = classify_response(
                no_mark, yes_mark, "NO", "YES"
            )

            rows.append({
                "label": label,
                "response": response,
                "no_mark_detected": no_mark,
                "yes_mark_detected": yes_mark,
            })

        extracted[table_name] = rows


def extract_mapped_tables(image, extracted, field_data):
    for output_name, mappings in FIELD_MAPPINGS.items():
        rows = []

        for section_id, field_id, label, first_key, second_key in mappings:
            field = find_field(field_data, section_id, field_id)
            regions = field.get("regions", {})

            first_region = regions.get(first_key)
            second_region = regions.get(second_key)

            if not first_region or not second_region:
                raise ValueError(
                    f"Missing checkbox regions for {section_id}.{field_id}"
                )

            first_mark = detect_region_mark(image, first_region)
            second_mark = detect_region_mark(image, second_region)

            response = classify_response(
                first_mark,
                second_mark,
                first_key.upper(),
                second_key.upper(),
            )

            rows.append({
                "label": label,
                "response": response,
                f"{first_key}_mark_detected": first_mark,
                f"{second_key}_mark_detected": second_mark,
            })

        extracted[output_name] = rows


def crop_mapped_region(image, region):
    """Crop a mapped text field using the original form coordinates."""

    scale_x = image.shape[1] / 612
    scale_y = image.shape[0] / 936

    left = max(0, round(region["x0"] * scale_x))
    right = min(image.shape[1], round(region["x1"] * scale_x))
    top = max(0, round(region["y0"] * scale_y))
    bottom = min(image.shape[0], round(region["y1"] * scale_y))

    crop = image[top:bottom, left:right]

    if crop.size == 0:
        return None

    return cv2.resize(
        crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )



def recognize_mapped_text(image, region, ocr_engine):
    """Recognize handwritten text inside one mapped region."""

    crop = crop_mapped_region(image, region)

    if crop is None:
        return {
            "value": None,
            "confidence": None,
            "status": "NEEDS_REVIEW",
        }

    # PaddleOCR requires a 3-channel image.
    if len(crop.shape) == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    predictions = ocr_engine.predict(
        crop,
        return_word_box=False,
    )

    recognized_text = []
    confidences = []

    for prediction in predictions:
        raw_data = prediction.json

        if callable(raw_data):
            raw_data = raw_data()

        if isinstance(raw_data, str):
            raw_data = json.loads(raw_data)

        data = raw_data.get("res", raw_data)

        texts = data.get("rec_texts", []) or []
        scores = data.get("rec_scores", []) or []

        for index, text in enumerate(texts):
            text = str(text).strip()

            if text:
                recognized_text.append(text)

                if index < len(scores):
                    score = float(scores[index])
                    confidences.append(
                        round(score * 100 if score <= 1 else score, 2)
                    )

    value = " ".join(recognized_text).strip()

    confidence = (
        round(sum(confidences) / len(confidences), 2)
        if confidences else None
    )

    status = (
        "HIGH" if confidence is not None and confidence >= 90
        else "MODERATE" if confidence is not None and confidence >= 70
        else "NEEDS_REVIEW"
    )

    return {
        "value": value or None,
        "confidence": confidence,
        "status": status,
    }



def extract_girls_fields(image, field_data, ocr_engine):
    """Extract Section D menstruation fields for girls."""

    results = {}

    for field_id, label, field_type in GIRLS_TEXT_FIELDS:
        field = find_field(field_data, "for_girls", field_id)
        region = field.get("regions", {}).get("value")

        if not region:
            raise ValueError(
                f"Missing value region for for_girls.{field_id}"
            )

        recognized = recognize_mapped_text(
            image,
            region,
            ocr_engine,
        )

        results[field_id] = {
            "label": label,
            "type": field_type,
            **recognized,
            "review_required": recognized["status"] == "NEEDS_REVIEW",
            "review_reason": (
                "No reliable text was recognized."
                if recognized["status"] == "NEEDS_REVIEW"
                else None
            ),
        }

    for field_id, definition in GIRLS_CHOICE_FIELDS.items():
        field = find_field(field_data, "for_girls", field_id)
        regions = field.get("regions", {})

        detected_options = []

        for option_key, option_label in definition["options"].items():
            region = regions.get(option_key)

            if not region:
                raise ValueError(
                    f"Missing region for for_girls.{field_id}.{option_key}"
                )

            if detect_region_mark(image, region):
                detected_options.append(option_label)

        if len(detected_options) == 1:
            response = detected_options[0]
            review_required = False
            review_reason = None
        elif len(detected_options) == 0:
            response = "UNMARKED"
            review_required = True
            review_reason = "No option was detected."
        else:
            response = "NEEDS_REVIEW"
            review_required = True
            review_reason = "Multiple options were detected."

        results[field_id] = {
            "label": definition["label"],
            "response": response,
            "detected_options": detected_options,
            "review_required": review_required,
            "review_reason": review_reason,
        }

    return results


def extract_history_tables(image, ocr_engine=None):
    """Extract CMDI Health Record Sections A-E and girls' menstruation data."""

    if image is None:
        raise ValueError("An image is required.")

    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")

    if image.size == 0:
        raise ValueError("The supplied image is empty.")

    if image.ndim not in (2, 3):
        raise ValueError("Unsupported image dimensions.")

    if image.ndim == 3:
        channels = image.shape[2]

        if channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Expected 3 or 4 image channels.")

    minimum_width = 1530
    minimum_height = 2340

    if (
        image.shape[1] < minimum_width
        or image.shape[0] < minimum_height
    ):
        raise ValueError(
            "Image resolution is below the recommended minimum "
            f"of {minimum_width}x{minimum_height}. "
            f"Received {image.shape[1]}x{image.shape[0]}."
        )

    if (
        image.shape[1] != REFERENCE_WIDTH
        or image.shape[0] != REFERENCE_HEIGHT
    ):
        image = cv2.resize(
            image,
            (REFERENCE_WIDTH, REFERENCE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    extracted = {}

    extract_original_tables(image, extracted)

    field_data = load_health_record_fields()
    extract_mapped_tables(image, extracted, field_data)

    if ocr_engine is not None:
        extracted["for_girls"] = extract_girls_fields(
            image,
            field_data,
            ocr_engine,
        )

    return extracted
