"""Static mapping of every source document to its department and access level."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentMeta:
    department: str   # hr | finance | exec
    access_level: int  # 1..3


# Keyed by filename (basename only — department folder is separate).
DOCUMENT_METADATA: dict[str, DocumentMeta] = {
    "leave_policy.pdf":          DocumentMeta(department="hr",      access_level=1),
    "code_of_conduct.pdf":       DocumentMeta(department="hr",      access_level=1),
    "performance_review.pdf":    DocumentMeta(department="hr",      access_level=2),
    "expense_policy.pdf":        DocumentMeta(department="finance",  access_level=1),
    "travel_reimbursement.pdf":  DocumentMeta(department="finance",  access_level=1),
    "compensation_committee.pdf":DocumentMeta(department="exec",     access_level=3),
    "strategic_plan.pdf":        DocumentMeta(department="exec",     access_level=3),
}


def get_metadata(filename: str) -> DocumentMeta:
    """Return metadata for *filename* (basename). Raises KeyError if unknown."""
    return DOCUMENT_METADATA[filename]
