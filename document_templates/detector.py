from document_templates.registry import TEMPLATE_REGISTRY


def detect_template(ocr_text):
    """
    Identify the document template using OCR-extracted text.
    """

    if not ocr_text:
        return None

    normalized_text = ocr_text.lower()

    # CMDI Health Record detection keywords
    cmdi_keywords = [
        "health record",
        "medical history",
        "physical examination",
        "for girls",
        "chief complaint",
    ]

    matches = sum(
        1 for keyword in cmdi_keywords
        if keyword in normalized_text
    )

    # Require at least two matching keywords
    if matches >= 2:
        return "cmdi_health_record"

    return None