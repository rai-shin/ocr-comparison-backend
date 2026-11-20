
from document_templates.cmdi_health_record.template import (
    extract_structured_data as extract_cmdi_health_record,
)


TEMPLATE_REGISTRY = {
    "cmdi_health_record": {
        "name": "CMDI Health Record",
        "extractor": extract_cmdi_health_record,
    },
}


def get_template(template_id):
    """Return a registered template by its ID."""

    return TEMPLATE_REGISTRY.get(template_id)


def get_available_templates():
    """Return the list of registered templates."""

    return [
        {
            "template_id": template_id,
            "template_name": template["name"],
        }
        for template_id, template in TEMPLATE_REGISTRY.items()
    ]
