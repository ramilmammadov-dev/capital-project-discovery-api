"""Structured-output schema for project extraction.

Designed for *strict* JSON-schema decoding (Claude ``output_config.format`` / OpenAI ``strict``):
every property is required and nothing is nullable, so the schema stays within strict-mode limits
(no ``anyOf``). Missing information is an empty string / empty list, and values are copied
verbatim — numbers, years and units are normalized afterwards by deterministic code.
"""

from __future__ import annotations

from typing import Any

from capex.models import BUDGET_TYPES, DOMAINS, FUNDING_CODES, PHASE_CODES

_PROJECT_PROPERTIES: dict[str, Any] = {
    "project_id": {
        "type": "string",
        "description": "Project identifier exactly as printed (project/CIP/TIP/job number); empty string if none.",
    },
    "project_name": {"type": "string", "description": "Official project name or title exactly as printed."},
    "project_description": {
        "type": "string",
        "description": "Scope and useful detail from the text: purpose, assets, equipment, facilities, "
        "infrastructure, requirements, budgeted scope (max ~400 characters); empty if none.",
    },
    "domain": {"type": "string", "enum": DOMAINS},
    "project_type": {
        "type": "string",
        "description": "Specific category, e.g. 'pavement preservation', 'interchange reconstruction', "
        "'water main replacement', 'fire station construction', 'HVAC/boiler replacement'.",
    },
    "location": {
        "type": "string",
        "description": "City, county, facility, roadway/segment/milepost or service area as printed; empty if none.",
    },
    "budget_text": {
        "type": "string",
        "description": "The project's cost or budget figure exactly as printed (prefer total project cost, "
        "otherwise the total shown for the project); empty if none. Do not compute sums.",
    },
    "budget_basis": {
        "type": "string",
        "description": "What budget_text represents, e.g. 'total project cost', 'FY2027 appropriation', "
        "'5-year total FY2027-2031'; empty if no budget.",
    },
    "budget_type": {"type": "string", "enum": BUDGET_TYPES},
    "funding_by_year": {
        "type": "array",
        "description": "Non-empty yearly amounts printed for this project.",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["year_label", "amount_text"],
            "properties": {
                "year_label": {"type": "string", "description": "Year column label as printed, e.g. 'FY 2027', '2026-27'."},
                "amount_text": {"type": "string", "description": "Amount exactly as printed."},
            },
        },
    },
    "funding_sources": {
        "type": "string",
        "description": "Funding sources as printed (e.g. 'Federal Aid, HURF, GO Bonds'); empty if none.",
    },
    "start_year_text": {
        "type": "string",
        "description": "Text giving the expected project/design start (e.g. 'FY 2027', 'Design: 2026'); empty if not stated.",
    },
    "construction_year_text": {
        "type": "string",
        "description": "Text giving the expected construction year; empty if not stated.",
    },
    "completion_year_text": {
        "type": "string",
        "description": "Text giving the expected completion; empty if not stated.",
    },
    "current_phase": {"type": "string", "enum": list(PHASE_CODES)},
    "phase_evidence": {
        "type": "string",
        "description": "Exact words from the text that support current_phase; empty if unknown.",
    },
    "phase_is_explicit": {
        "type": "boolean",
        "description": "true only when the text explicitly states the project's stage/status.",
    },
    "status_text": {
        "type": "string",
        "description": "Specific status wording as printed (e.g. 'Deferred to FY2030', 'Under construction'); empty if none.",
    },
    "funding_status": {"type": "string", "enum": list(FUNDING_CODES)},
    "funding_evidence": {
        "type": "string",
        "description": "Exact words supporting funding_status; empty if unknown.",
    },
    "source_pages": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "PDF page number(s) from the [[PDF page N]] marker(s) above this project's entry.",
    },
    "source_section": {"type": "string", "description": "Heading, table or section that contains the entry."},
    "source_excerpt": {
        "type": "string",
        "description": "Short verbatim quotation (<= 200 characters) copied from CONTENT that identifies this project.",
    },
    "extraction_confidence": {
        "type": "number",
        "description": "0 to 1: confidence that this is a genuine capital project and the fields are correct.",
    },
}

PROJECT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(_PROJECT_PROPERTIES),
    "properties": _PROJECT_PROPERTIES,
}

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["amount_unit", "projects"],
    "properties": {
        "amount_unit": {
            "type": "string",
            "enum": ["dollars", "thousands", "millions", "unknown"],
            "description": "Unit of the money figures in CONTENT, e.g. 'thousands' when a header or note says "
            "($000), (000s), 'in thousands'.",
        },
        "projects": {"type": "array", "items": PROJECT_SCHEMA},
    },
}
