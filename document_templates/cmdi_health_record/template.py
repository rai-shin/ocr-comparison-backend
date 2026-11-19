from document_templates.cmdi_health_record.history_extraction import (
    extract_history_tables,
)


TEMPLATE_ID = "cmdi_health_record"
TEMPLATE_NAME = "CMDI Health Record"


def add_review_information(rows):
    """
    Add manual-review information without changing
    the original extracted response or detection results.
    """

    enriched_rows = []

    for row in rows:
        response = row["response"]

        if response == "NEEDS_REVIEW":
            review_required = True
            review_reason = "Both NO and YES marks were detected."

        elif response == "UNMARKED":
            review_required = True
            review_reason = (
                "No mark was detected. Confirm whether the field "
                "was intentionally left blank."
            )

        else:
            review_required = False
            review_reason = None

        enriched_rows.append({
            **row,
            "review_required": review_required,
            "review_reason": review_reason,
        })

    return enriched_rows


def extract_structured_data(image):
    """
    Extract structured data specific to the CMDI Health Record.

    Raw OCR results are handled separately and are not modified here.
    """

    history_tables = extract_history_tables(image)

    return {
        "template_id": TEMPLATE_ID,
        "template_name": TEMPLATE_NAME,
        "structured_data": {
            "medical_history": add_review_information(
                history_tables["medical_history"]
            ),
            "family_history": add_review_information(
                history_tables["family_history"]
            ),
        },
    }
