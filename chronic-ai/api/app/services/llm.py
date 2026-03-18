"""
LLM Service.
"""
import asyncio
from typing import Any, AsyncGenerator, Optional
from uuid import UUID
import base64
import hashlib
import json
import logging
import math
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
import uuid

from pydantic import ValidationError

from app.services.ecg_classifier_service import ecg_classifier_service
from app.services.cache import cache_response, get_cached_response, response_cache
from app.services.llm_client import llm_client
from app.services.rag import get_patient_context
from app.config import settings
from app.models.schemas import MedicalRecordAIAnalysis

logger = logging.getLogger(__name__)

# System prompt for MedGemma medical reasoning
MEDICAL_REASONING_SYSTEM = """You are a helpful medical AI assistant for Vietnamese healthcare.
You assist doctors and patients with chronic disease management.
Use the provided patient context to give accurate, personalized responses.

IMPORTANT GUIDELINES:
- Be thorough but concise in your explanations
- Always consider the patient's medical history and current medications
- Flag any potential drug interactions or contraindications
- For serious symptoms, recommend seeking immediate medical attention
- Explain medical concepts in simple terms when addressing patients
- Include relevant warnings about symptoms that require urgent care

CRITICAL - HANDLING MISSING DATA:
- NEVER output placeholder text like [Insert...], [TODO], [N/A], [Date unknown], etc.
- NEVER use bracket notation to indicate missing information
- If specific data is not available in the provided context, state it naturally
- Example: Instead of "[Insert Last Checkup Date]", say "This information is not available in your records"
- Only answer based on information actually present in the patient context
- If important data is missing, suggest checking with the healthcare provider

Remember: You are a support tool, not a replacement for professional medical advice."""

UPLOAD_ANALYSIS_SYSTEM = """You are a clinical decision-support assistant for doctors.
Analyze uploaded medical records and produce concise, practical insights.
Return plain text using the exact section structure requested by the user prompt.
Do not return JSON, markdown fences, or extra commentary outside the requested sections."""

UPLOAD_ANALYSIS_CACHE_TYPE = "upload_analysis:v4"

UPLOAD_ANALYSIS_SECTIONED_FORMAT = """Return plain text using exactly this structure:

Summary: <short clinical summary>
Key findings:
- <finding 1>
- <finding 2>
Clinical significance: <why this matters clinically>
Recommended follow-up:
- <follow-up action 1>
- <follow-up action 2>
Urgency: low|medium|high
Confidence: low|medium|high
Limitations:
- <known uncertainty or missing data>
"""


def _resolve_upload_analysis_model(*, has_image: bool) -> str:
    """Choose model for upload analysis, preferring dedicated image-analysis model."""
    if has_image:
        configured = (settings.upload_analysis_model or "").strip()
        if configured:
            return configured
    return settings.medical_model


def _multimodal_route_capability(*, route_name: str, model: str) -> tuple[bool, str]:
    """Return whether a route is allowed to send images to the configured LLM path."""
    if route_name == "upload_analysis" and not settings.enable_multimodal_upload_analysis:
        return False, "Upload analysis multimodal requests are disabled by configuration."
    if route_name == "medical_reasoning" and not settings.enable_multimodal_medical_reasoning:
        return False, "Medical reasoning multimodal requests are disabled by configuration."
    return llm_client.can_use_images(model)


def _strip_markdown_code_fence(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""

    fenced_match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()
    return text


def _extract_first_balanced_json_object(raw_text: str) -> Optional[str]:
    text = str(raw_text or "")
    if not text:
        return None

    in_string = False
    escape = False
    depth = 0
    start_index: Optional[int] = None

    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                return text[start_index:index + 1]

    return None


def _extract_json_object(raw_text: str) -> Optional[dict[str, Any]]:
    """Extract first JSON object from a model response."""
    if not raw_text:
        return None

    candidates: list[str] = []
    for candidate in [
        str(raw_text or "").strip(),
        _strip_markdown_code_fence(raw_text),
    ]:
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
        balanced_candidate = _extract_first_balanced_json_object(candidate)
        if balanced_candidate and balanced_candidate not in candidates:
            candidates.append(balanced_candidate)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            continue

    return None


def _normalize_label_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[_*`#>\-]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _canonical_upload_analysis_field(label: str) -> Optional[str]:
    normalized = _normalize_label_text(label)
    field_aliases = {
        "summary": {"summary", "tom tat"},
        "key_findings": {"key findings", "findings", "diem chinh", "phat hien"},
        "clinical_significance": {
            "clinical significance",
            "significance",
            "y nghia lam sang",
        },
        "recommended_follow_up": {
            "recommended follow up",
            "follow up",
            "followup",
            "de xuat theo doi",
            "khuyen nghi",
        },
        "urgency": {"urgency", "muc do khan", "khan cap"},
        "confidence": {"confidence", "do tin cay"},
        "limitations": {"limitations", "gioi han"},
    }
    for field_name, aliases in field_aliases.items():
        if normalized in aliases:
            return field_name
    return None


def _to_string_list(value: Any, max_items: int = 5) -> list[str]:
    """Normalize a model field into a list of non-empty strings."""
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [value]
    else:
        return []

    cleaned: list[str] = []
    for item in items:
        item_str = str(item).strip()
        if item_str:
            cleaned.append(item_str)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _sanitize_text(value: Any, max_len: int = 1200) -> str:
    """
    Normalize text for safe DB JSON storage.

    Removes null/control chars that commonly break JSONB insertion.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    # Remove null bytes first.
    text = text.replace("\x00", " ")
    # Keep printable chars and common whitespace.
    text = "".join(ch for ch in text if ch == "\n" or ch == "\r" or ch == "\t" or ord(ch) >= 32)
    # Drop invalid unicode code points for storage safety.
    text = text.encode("utf-8", "ignore").decode("utf-8")
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def _sanitize_list(values: list[str], max_items: int = 5, max_item_len: int = 400) -> list[str]:
    """Sanitize a list of strings for storage safety."""
    out: list[str] = []
    for value in values[:max_items]:
        cleaned = _sanitize_text(value, max_len=max_item_len)
        if cleaned:
            out.append(cleaned)
    return out


def _coerce_upload_analysis_list(
    value: Any,
    *,
    max_items: int = 5,
    max_item_len: int = 400,
) -> list[str]:
    """Best-effort normalization for list-like LLM fields."""
    if isinstance(value, list):
        return _sanitize_list([str(item) for item in value], max_items=max_items, max_item_len=max_item_len)

    if not isinstance(value, str):
        return []

    text = _sanitize_text(value, max_len=max_items * max_item_len)
    if not text:
        return []

    parts = [
        part.strip(" -•*;\t")
        for part in re.split(r"\n+|;\s+|(?<!\d)\s+[•*-]\s+|\s*\|\s*", text)
        if part.strip(" -•*;\t")
    ]
    if len(parts) <= 1:
        numbered_parts = [
            item.strip(" -•*;\t")
            for item in re.split(r"\s*(?:\d+[.)])\s*", text)
            if item.strip(" -•*;\t")
        ]
        if len(numbered_parts) > 1:
            parts = numbered_parts

    return _sanitize_list(parts or [text], max_items=max_items, max_item_len=max_item_len)


def _extract_plain_text_summary(raw_text: str) -> str:
    """
    Build a human-readable summary fallback from model output.

    Avoid returning raw JSON blobs or markdown fences as summary text.
    """
    normalized = _strip_markdown_code_fence(raw_text)
    if not normalized:
        return ""

    cleaned_lines: list[str] = []
    for line in normalized.splitlines():
        text = line.strip().strip(",")
        if not text or text in {"{", "}", "[", "]"}:
            continue
        if text.startswith("```"):
            continue
        if re.match(r'^"[^"]+"\s*:', text):
            continue
        cleaned_lines.append(text)

    cleaned = " ".join(cleaned_lines).strip() if cleaned_lines else normalized.strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return ""
    if cleaned.startswith("{") or cleaned.startswith("["):
        return ""

    return _sanitize_text(cleaned, max_len=500)


def _looks_like_garbled_summary(text: str) -> bool:
    """
    Detect low-quality raw model text that should not be shown as a summary.

    This is only used for fallback plain-text extraction when the model failed
    to return valid JSON. The goal is to avoid surfacing token soup such as
    repeated numbers/connectors ("25, and, and, and") in the UI.
    """
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not normalized:
        return True

    tokens = re.findall(r"[a-zA-ZÀ-Ỵà-ỵ0-9]+", normalized)
    if len(tokens) < 6:
        return False

    unique_ratio = len(set(tokens)) / len(tokens)
    top_token_count = max(tokens.count(token) for token in set(tokens))
    repeated_bigrams = 0
    seen_bigrams: set[tuple[str, str]] = set()
    for idx in range(len(tokens) - 1):
        bigram = (tokens[idx], tokens[idx + 1])
        if bigram in seen_bigrams:
            repeated_bigrams += 1
        else:
            seen_bigrams.add(bigram)

    return (
        unique_ratio < 0.35
        or top_token_count >= max(5, math.ceil(len(tokens) * 0.3))
        or repeated_bigrams >= max(3, math.ceil((len(tokens) - 1) * 0.25))
    )


def _resolve_upload_analysis_summary(
    parsed: dict[str, Any],
    raw_text: str,
    *,
    fallback_message: str,
) -> str:
    summary = _sanitize_text(parsed.get("summary"), max_len=1200)
    if summary:
        return summary

    fallback_summary = _extract_plain_text_summary(raw_text)
    if fallback_summary and not _looks_like_garbled_summary(fallback_summary):
        return fallback_summary

    return fallback_message


def _preview_model_output_for_log(raw_text: str, max_len: int = 280) -> str:
    """Create a short one-line preview of raw model output for debugging logs."""
    normalized = _strip_markdown_code_fence(raw_text)
    preview = re.sub(r"\s+", " ", normalized).strip()
    if not preview:
        return "<empty>"
    if len(preview) > max_len:
        return f"{preview[:max_len].rstrip()}..."
    return preview


def _extract_structured_sections_from_text(raw_text: str) -> dict[str, Any]:
    """
    Best-effort section extraction for weak model outputs that are not valid JSON.

    Supports simple markdown/text formats such as:
    Summary: ...
    Key findings:
    - item 1
    - item 2
    """
    normalized = _strip_markdown_code_fence(raw_text)
    if not normalized:
        return {}

    sections: dict[str, list[str]] = {}
    current_field: Optional[str] = None
    fallback_lines: list[str] = []

    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        candidate = line.lstrip("-*•# ").strip()
        if not candidate:
            continue

        label_part: Optional[str] = None
        content_part = ""
        for separator in (":", "-", " - "):
            if separator in candidate:
                possible_label, possible_content = candidate.split(separator, 1)
                canonical = _canonical_upload_analysis_field(possible_label)
                if canonical:
                    label_part = canonical
                    content_part = possible_content.strip()
                    break

        if not label_part:
            canonical = _canonical_upload_analysis_field(candidate)
            if canonical:
                label_part = canonical
                content_part = ""

        if label_part:
            current_field = label_part
            sections.setdefault(current_field, [])
            if content_part:
                sections[current_field].append(content_part)
            continue

        if current_field:
            sections.setdefault(current_field, []).append(candidate)
        else:
            fallback_lines.append(candidate)

    extracted: dict[str, Any] = {}

    summary_candidates = sections.get("summary") or []
    summary_text = " ".join(summary_candidates).strip()
    if not summary_text and fallback_lines:
        summary_text = fallback_lines[0]
    summary_text = _sanitize_text(summary_text, max_len=1200)
    if summary_text and not _looks_like_garbled_summary(summary_text):
        extracted["summary"] = summary_text

    key_findings = _coerce_upload_analysis_list("\n".join(sections.get("key_findings") or []))
    if key_findings:
        extracted["key_findings"] = key_findings

    clinical_significance = _sanitize_text(
        " ".join(sections.get("clinical_significance") or []),
        max_len=1200,
    )
    if clinical_significance:
        extracted["clinical_significance"] = clinical_significance

    follow_up = _coerce_upload_analysis_list("\n".join(sections.get("recommended_follow_up") or []))
    if follow_up:
        extracted["recommended_follow_up"] = follow_up

    limitations = _coerce_upload_analysis_list("\n".join(sections.get("limitations") or []), max_item_len=500)
    if limitations:
        extracted["limitations"] = limitations

    urgency_text = " ".join(sections.get("urgency") or []).strip()
    if urgency_text:
        extracted["urgency"] = urgency_text

    confidence_text = " ".join(sections.get("confidence") or []).strip()
    if confidence_text:
        extracted["confidence"] = confidence_text

    return extracted


def _normalize_upload_analysis_level(value: Any, default: str = "medium") -> str:
    normalized = _normalize_label_text(str(value or ""))
    if normalized in {"high", "cao"}:
        return "high"
    if normalized in {"low", "thap"}:
        return "low"
    if normalized in {"medium", "trung binh", "moderate"}:
        return "medium"
    return default


def _repair_upload_analysis_payload(parsed: Optional[dict[str, Any]], raw_text: str) -> dict[str, Any]:
    """Repair weak model output into the expected upload-analysis shape."""
    extracted = _extract_structured_sections_from_text(raw_text)
    source = dict(parsed or {})

    for alias, canonical in {
        "findings": "key_findings",
        "clinical_assessment": "clinical_significance",
        "significance": "clinical_significance",
        "follow_up": "recommended_follow_up",
        "recommended_actions": "recommended_follow_up",
        "next_steps": "recommended_follow_up",
    }.items():
        if canonical not in source and alias in source:
            source[canonical] = source.get(alias)

    repaired: dict[str, Any] = {}

    summary = _sanitize_text(
        source.get("summary") or extracted.get("summary"),
        max_len=1200,
    )
    if summary and not _looks_like_garbled_summary(summary):
        repaired["summary"] = summary

    key_findings = _coerce_upload_analysis_list(
        source.get("key_findings") or extracted.get("key_findings"),
    )
    if key_findings:
        repaired["key_findings"] = key_findings

    clinical_significance = _sanitize_text(
        source.get("clinical_significance") or extracted.get("clinical_significance"),
        max_len=1200,
    )
    if clinical_significance:
        repaired["clinical_significance"] = clinical_significance

    recommended_follow_up = _coerce_upload_analysis_list(
        source.get("recommended_follow_up") or extracted.get("recommended_follow_up"),
    )
    if recommended_follow_up:
        repaired["recommended_follow_up"] = recommended_follow_up

    limitations = _coerce_upload_analysis_list(
        source.get("limitations") or extracted.get("limitations"),
        max_item_len=500,
    )
    if limitations:
        repaired["limitations"] = limitations

    if "urgency" in source or "urgency" in extracted:
        repaired["urgency"] = _normalize_upload_analysis_level(
            source.get("urgency") or extracted.get("urgency"),
        )

    if "confidence" in source or "confidence" in extracted:
        repaired["confidence"] = _normalize_upload_analysis_level(
            source.get("confidence") or extracted.get("confidence"),
        )

    return repaired


def _repaired_payload_has_meaningful_content(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("summary")
        or payload.get("key_findings")
        or payload.get("clinical_significance")
        or payload.get("recommended_follow_up")
    )


def _validated_upload_analysis_result(
    result: dict[str, Any],
    *,
    fallback_message: str,
) -> dict[str, Any]:
    """Validate a structured upload-analysis payload before caching or returning it."""
    try:
        validated = MedicalRecordAIAnalysis.model_validate(result)
        return validated.model_dump(mode="json", by_alias=True, exclude_none=True)
    except ValidationError as exc:
        logger.warning(
            "[upload-analysis] structured payload validation failed request_id=%s errors=%s",
            result.get("request_id"),
            exc.errors(),
        )
        fallback_result = {
            "model": result.get("model"),
            "record_type": result.get("record_type"),
            "generated_at": result.get("generated_at"),
            "request_id": result.get("request_id"),
            "status": "error",
            "summary": fallback_message,
            "key_findings": [],
            "recommended_follow_up": [],
            "limitations": ["LLM returned invalid structured output."],
        }
        validated = MedicalRecordAIAnalysis.model_validate(fallback_result)
        return validated.model_dump(mode="json", by_alias=True, exclude_none=True)


def _scores_are_probability_like(scores: list[float]) -> bool:
    return bool(scores) and all(0.0 <= score <= 1.0 for score in scores)


def _softmax_normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    if _scores_are_probability_like(scores):
        return [float(score) for score in scores]

    max_score = max(scores)
    weights = [math.exp(score - max_score) for score in scores]
    total = sum(weights)
    if total <= 0:
        return [0.0 for _ in scores]
    return [weight / total for weight in weights]


_PATIENT_SUMMARY_SECTION_HEADERS: list[tuple[str, str]] = [
    (r"(?:Danh sách vấn đề(?:\s*\(Problem List\))?|Problem List)", "Danh sách vấn đề (Problem List)"),
    (
        r"(?:Thuốc đang dùng(?:\s*\(Current Medications\))?|Current Medications)",
        "Thuốc đang dùng (Current Medications)",
    ),
    (r"(?:Dị ứng(?:\s*\(Allergies\))?|Allergies)", "Dị ứng (Allergies)"),
    (
        r"(?:Diễn tiến bệnh(?:\s*\(Disease Progress\))?|Disease Progress)",
        "Diễn tiến bệnh (Disease Progress)",
    ),
    (
        r"(?:Tóm tắt sinh hiệu gần nhất(?:\s*\(Recent Vitals\))?|Recent Vitals)",
        "Tóm tắt sinh hiệu gần nhất (Recent Vitals)",
    ),
    (
        r"(?:Đánh giá lâm sàng(?:\s*\(Clinical Assessment\))?|Clinical Assessment)",
        "Đánh giá lâm sàng (Clinical Assessment)",
    ),
]


def _strip_unbalanced_double_asterisks(text: str) -> str:
    """Remove broken bold markers on lines with odd '**' pairs."""
    normalized_lines: list[str] = []
    for line in text.split("\n"):
        if line.count("**") % 2 != 0:
            normalized_lines.append(line.replace("**", ""))
        else:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)


def _break_dense_lines(text: str) -> str:
    """Split long dense lines into short paragraphs for markdown rendering."""
    lines = text.split("\n")
    output: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            output.append("")
            continue

        if (
            len(line) < 170
            or line.startswith("## ")
            or re.match(r"^[-*]\s", line)
            or re.match(r"^\d+[.)]\s", line)
        ):
            output.append(line)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", line)
        paragraph: list[str] = []
        paragraph_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            starts_new = (
                re.match(r"^\d+[.)]\s", sentence)
                or re.match(r"^[-*]\s", sentence)
                or sentence.startswith("## ")
                or paragraph_len > 250
            )

            if starts_new and paragraph:
                output.append(" ".join(paragraph))
                output.append("")
                paragraph = []
                paragraph_len = 0

            paragraph.append(sentence)
            paragraph_len += len(sentence) + 1

        if paragraph:
            output.append(" ".join(paragraph))

    cleaned: list[str] = []
    prev_blank = False
    for line in output:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    while cleaned and cleaned[-1].strip() == "":
        cleaned.pop()
    return "\n".join(cleaned)


def _normalize_patient_summary_markdown(text: str) -> str:
    """
    Normalize patient profile summary markdown to avoid inline wall-of-text output.

    Handles malformed section headers, run-on numbered lists, and broken bold markers.
    """
    summary = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not summary:
        return ""

    summary = summary.replace("\u00a0", " ")
    summary = re.sub(r"[ \t]+", " ", summary)
    summary = _strip_unbalanced_double_asterisks(summary)

    # Fix missing space after sentence-ending punctuation.
    summary = re.sub(r"([.!?])(?=[A-ZÀ-Ỵa-zà-ỵ#*\\[])", r"\1 ", summary)

    # Promote known sections to consistent markdown headers.
    for pattern, canonical_header in _PATIENT_SUMMARY_SECTION_HEADERS:
        summary = re.sub(
            rf"(?:(?<=^)|(?<=[\n.!?]))\s*(?:#{1,4}\s*)?(?:\*\*)?\s*(?:{pattern})\s*(?:\*\*)?\s*:?\s*",
            f"\n\n## {canonical_header}\n",
            summary,
            flags=re.IGNORECASE,
        )

    # Ensure headers always start on their own block.
    summary = re.sub(r"(?<!\n)(##\s)", r"\n\n\1", summary)
    summary = re.sub(r"(##[^\n]+)\s*(?=(?:\d+[.)]|[-•*]))", r"\1\n", summary)

    # Normalize numbered lists (e.g., "1.[I10]" or "... điều trị.2.[E78.5]").
    summary = re.sub(r"(?<![A-Za-zÀ-Ỵa-zà-ỵ0-9])(\d+[.)])(?=\S)", r"\1 ", summary)
    summary = re.sub(r"(?<!^)(?<!\n)(\d+[.)]\s*(?=[\[A-ZÀ-Ỵa-zà-ỵ]))", r"\n\1", summary)

    # Normalize malformed inline bullets to markdown list items.
    summary = re.sub(r"([:;.!?\n])\s*\*(?=[A-ZÀ-Ỵa-zà-ỵ0-9])", r"\1\n- ", summary)
    summary = re.sub(r"([:;.!?\n])\s*([•●▪])\s*(?=[A-ZÀ-Ỵa-zà-ỵ0-9])", r"\1\n- ", summary)
    summary = re.sub(r"([:;.!?])\s*-\s+(?=[A-ZÀ-Ỵa-zà-ỵ0-9])", r"\1\n- ", summary)

    # Keep only the first occurrence of each canonical section header.
    canonical_titles = {
        re.sub(r"\s+", " ", header).strip().lower()
        for _, header in _PATIENT_SUMMARY_SECTION_HEADERS
    }
    seen_titles: set[str] = set()
    deduped_lines: list[str] = []
    for line in summary.split("\n"):
        stripped = line.strip()
        match = re.match(r"^##\s+(.+?)\s*:?\s*$", stripped, flags=re.IGNORECASE)
        if match:
            normalized_title = re.sub(r"\s+", " ", match.group(1)).strip().lower()
            if normalized_title in canonical_titles:
                if normalized_title in seen_titles:
                    continue
                seen_titles.add(normalized_title)
        deduped_lines.append(line)
    summary = "\n".join(deduped_lines)

    summary = re.sub(r"\n{3,}", "\n\n", summary).strip()
    return _break_dense_lines(summary)


def _top_scores_for_log(scores_by_class: dict[str, float], top_k: int = 3) -> list[tuple[str, float]]:
    ranked = sorted(
        ((str(label), float(score)) for label, score in scores_by_class.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:top_k]


ECG_CLASS_DESCRIPTIONS: dict[str, str] = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}

ECG_CLASS_DESCRIPTIONS_VI: dict[str, str] = {
    "NORM": "ECG trong giới hạn bình thường",
    "MI": "gợi ý nhồi máu cơ tim",
    "STTC": "thay đổi ST/T",
    "CD": "rối loạn dẫn truyền",
    "HYP": "dấu hiệu phì đại",
}

_ECG_LOW_QUALITY_OUTPUT_MARKERS = (
    "ecg image shows",
    "normal ecg",
    "no follow up actions",
    "urgency low medium high",
    "confidence low medium high",
)


def _format_percent(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _sorted_prediction_score_rows(
    prediction_scores: list[dict[str, Any]],
    *,
    top_k: Optional[int] = None,
) -> list[dict[str, Any]]:
    ranked = sorted(
        prediction_scores,
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    if top_k is None:
        return ranked
    return ranked[:top_k]


def _estimate_ecg_classifier_confidence(
    prediction_scores: list[dict[str, Any]],
) -> str:
    ranked = _sorted_prediction_score_rows(prediction_scores, top_k=2)
    if not ranked:
        return "low"
    top_score = float(ranked[0].get("score") or 0.0)
    second_score = float(ranked[1].get("score") or 0.0) if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    if top_score >= 0.8 and margin >= 0.45:
        return "high"
    if top_score >= 0.55 and margin >= 0.15:
        return "medium"
    return "low"


def _estimate_ecg_classifier_urgency(
    prediction_scores: list[dict[str, Any]],
    predicted_labels: list[str],
) -> str:
    ranked = _sorted_prediction_score_rows(prediction_scores, top_k=1)
    if not ranked:
        return "medium"

    predicted = {str(label).strip().upper() for label in (predicted_labels or []) if str(label).strip()}
    top_label = str(ranked[0].get("class") or "").upper()
    top_score = float(ranked[0].get("score") or 0.0)

    if "MI" in predicted or (top_label == "MI" and top_score >= 0.35):
        return "high"
    if predicted.intersection({"STTC", "CD", "HYP"}):
        return "medium"
    if top_label in {"STTC", "CD", "HYP"} and top_score >= 0.35:
        return "medium"
    if top_label == "NORM" and top_score >= 0.65:
        return "low"
    return "medium"


def _looks_like_low_quality_ecg_output(
    repaired_payload: dict[str, Any],
    raw_text: str,
) -> bool:
    normalized_raw = _normalize_label_text(raw_text)
    if any(marker in normalized_raw for marker in _ECG_LOW_QUALITY_OUTPUT_MARKERS):
        return True

    summary = _normalize_label_text(repaired_payload.get("summary") or "")
    findings = " ".join(
        _normalize_label_text(item)
        for item in (repaired_payload.get("key_findings") or [])
    ).strip()
    follow_up = " ".join(
        _normalize_label_text(item)
        for item in (repaired_payload.get("recommended_follow_up") or [])
    ).strip()

    placeholder_pairs = (
        ("ecg image shows", summary),
        ("normal ecg", summary),
        ("no follow up actions", follow_up),
    )
    if any(token in text for token, text in placeholder_pairs if text):
        return True

    if summary and findings and summary == findings:
        return True

    return False


def _build_ecg_classifier_fallback_analysis(
    *,
    prediction_scores: list[dict[str, Any]],
    predicted_labels: list[str],
    reason: str,
) -> dict[str, Any]:
    ranked = _sorted_prediction_score_rows(prediction_scores, top_k=3)
    if not ranked:
        return {
            "summary": "Không thể suy ra nhận định ECG đáng tin cậy từ bộ phân loại.",
            "key_findings": [],
            "clinical_significance": "Cần đọc ECG gốc bởi bác sĩ để kết luận.",
            "recommended_follow_up": [
                "Đọc lại ECG gốc và đối chiếu với triệu chứng lâm sàng.",
            ],
            "urgency": "medium",
            "confidence": "low",
            "limitations": [
                reason,
                "Không có đủ điểm phân loại để tạo tóm tắt tự động.",
            ],
        }

    top_row = ranked[0]
    top_label = str(top_row.get("class") or "").upper()
    top_score = float(top_row.get("score") or 0.0)
    top_vi = ECG_CLASS_DESCRIPTIONS_VI.get(top_label, ECG_CLASS_DESCRIPTIONS.get(top_label, top_label))
    top_en = ECG_CLASS_DESCRIPTIONS.get(top_label, top_label)
    alternatives = [
        f"{str(row.get('class') or '').upper()} {_format_percent(float(row.get('score') or 0.0))}"
        for row in ranked[1:]
        if row.get("class")
    ]
    predicted = [str(label).strip().upper() for label in (predicted_labels or []) if str(label).strip()]

    if top_label == "NORM":
        summary = (
            f"Bộ phân loại ECG tự động nghiêng về ECG bình thường, với nhóm {top_label} "
            f"({top_en}) cao nhất ở mức {_format_percent(top_score)}."
        )
    else:
        summary = (
            f"Bộ phân loại ECG tự động ưu tiên nhóm {top_label} ({top_vi}) với xác suất hiển thị "
            f"{_format_percent(top_score)}; cần đối chiếu ngay với ECG gốc và bệnh cảnh lâm sàng."
        )

    key_findings = [
        f"Nhóm điểm cao nhất: {top_label} ({top_en}) {_format_percent(top_score)}.",
    ]
    if alternatives:
        key_findings.append(f"Các nhóm tiếp theo: {', '.join(alternatives)}.")
    if predicted:
        key_findings.append(f"Nhãn vượt ngưỡng của bộ phân loại: {', '.join(predicted)}.")
    key_findings.append(
        "Kết quả này được suy ra từ bộ phân loại ảnh ECG và không thay thế cho đọc ECG chuẩn 12 chuyển đạo."
    )

    urgency = _estimate_ecg_classifier_urgency(prediction_scores, predicted)
    confidence = _estimate_ecg_classifier_confidence(prediction_scores)

    if urgency == "high":
        follow_up = [
            "Đánh giá tim mạch khẩn, đối chiếu ECG gốc và triệu chứng ngay.",
            "Nếu có đau ngực, khó thở, ngất hoặc huyết động không ổn định, xử trí cấp cứu ngay.",
        ]
    elif urgency == "medium":
        follow_up = [
            "Đọc lại ECG gốc, so sánh với triệu chứng và ECG trước đó nếu có.",
            "Cân nhắc hội chẩn tim mạch nếu còn nghi ngờ hoặc bệnh nhân có triệu chứng.",
        ]
    else:
        follow_up = [
            "Đối chiếu với triệu chứng hiện tại và bản ECG gốc trước khi kết luận.",
            "Nếu bệnh nhân có đau ngực, khó thở, ngất hoặc triệu chứng cấp, vẫn cần đánh giá y khoa sớm.",
        ]

    clinical_significance = (
        "Ưu tiên xem đây là gợi ý sàng lọc từ bộ phân loại ảnh; quyết định lâm sàng cần dựa trên ECG gốc, "
        "triệu chứng và thăm khám."
    )

    return {
        "summary": summary,
        "key_findings": _sanitize_list(key_findings, max_items=4, max_item_len=400),
        "clinical_significance": _sanitize_text(clinical_significance, max_len=1200),
        "recommended_follow_up": _sanitize_list(follow_up, max_items=3, max_item_len=400),
        "urgency": urgency,
        "confidence": confidence,
        "limitations": _sanitize_list(
            [
                reason,
                "Phản hồi từ LLM không đủ chất lượng nên phần diễn giải này được dựng từ điểm bộ phân loại.",
                "Không thay thế cho diễn giải ECG bởi bác sĩ.",
            ],
            max_items=4,
            max_item_len=500,
        ),
    }


def _classify_llm_error(message: str) -> str:
    """Convert low-level LLM errors into backend diagnostic reason text."""
    msg = (message or "").lower()
    if "notimplementederror" in msg:
        return "Server runtime does not support async subprocess execution for token retrieval."
    if "not found" in msg and "model" in msg:
        return "Configured model is unavailable. Please verify MEDICAL_MODEL in backend configuration."
    if "gcloud" in msg or "access token" in msg or "auth" in msg:
        return "Google Cloud authentication is unavailable for this server session."
    if "vertex ai error (401)" in msg or "vertex ai error (403)" in msg:
        return "The backend is not authorized to call the Vertex endpoint."
    if "timeout" in msg:
        return "The model request timed out."
    if "vertex ai error (400)" in msg or "invalid" in msg:
        return "The request payload was rejected by the model endpoint."
    if "cannot connect" in msg:
        return "The backend cannot reach the model endpoint."
    return "The model request failed unexpectedly."


def _sha256_hex(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _build_upload_analysis_cache_key(
    *,
    record_type: str,
    title: str,
    extracted_text: str,
    image_base64: Optional[str],
) -> str:
    """
    Build deterministic cache key for upload analysis reuse.
    """
    normalized_type = (record_type or "").strip().lower()
    normalized_title = re.sub(r"\s+", " ", (title or "").strip().lower())
    text_hash = _sha256_hex(extracted_text or "")
    image_hash = _sha256_hex(image_base64 or "")
    title_hash = _sha256_hex(normalized_title)
    return (
        f"type:{normalized_type}|title:{title_hash[:16]}|"
        f"text:{text_hash[:24]}|image:{image_hash[:24]}"
    )


def _decode_cached_upload_analysis(payload: str) -> Optional[dict[str, Any]]:
    """
    Decode cached upload analysis JSON payload.
    """
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    try:
        validated = MedicalRecordAIAnalysis.model_validate(data)
    except ValidationError:
        return None
    return validated.model_dump(mode="json", by_alias=True, exclude_none=True)


async def _get_cached_upload_analysis(cache_key: str) -> Optional[dict[str, Any]]:
    """
    Return cached upload analysis result if present.
    """
    if not response_cache.enabled:
        return None
    cached = await get_cached_response(cache_key, query_type=UPLOAD_ANALYSIS_CACHE_TYPE)
    if not cached:
        return None
    payload, _ = cached
    return _decode_cached_upload_analysis(payload)


async def _store_upload_analysis_cache(cache_key: str, result: dict[str, Any]) -> None:
    """
    Store successful upload analysis result for reuse.
    """
    if not response_cache.enabled:
        return
    if not isinstance(result, dict):
        return
    status = str(result.get("status") or "").lower()
    if status not in {"completed", "skipped"}:
        return
    try:
        serialized = json.dumps(result, ensure_ascii=False)
    except Exception:
        return
    await cache_response(
        query=cache_key,
        response=serialized,
        query_type=UPLOAD_ANALYSIS_CACHE_TYPE,
        metadata={"kind": "upload_analysis", "status": status},
    )


async def _analyze_ecg_with_classifier(
    *,
    request_id: str,
    record_type: str,
    safe_title: str,
    image_base64: str,
    analysis_model: str,
    base_result: dict[str, Any],
) -> dict[str, Any]:
    """
    ECG-only analysis flow:
    image -> MedSigLIP embedding -> classifier scores -> MedGemma final analysis.
    """
    start_total = time.perf_counter()
    logger.info(
        "[upload-analysis][ecg] start id=%s model=%s image_base64_len=%s",
        request_id,
        analysis_model,
        len(image_base64 or ""),
    )

    model_available = await llm_client.check_model_available(analysis_model)
    if not model_available:
        logger.error(
            "[upload-analysis][ecg] medical model unavailable id=%s model=%s",
            request_id,
            analysis_model,
        )
        raise RuntimeError(f"Model not available: {settings.medical_model}")

    logger.info("[upload-analysis][ecg] classifier inference start id=%s", request_id)
    start_classifier = time.perf_counter()
    classifier_output = await ecg_classifier_service.predict_from_base64(
        image_base64,
    )
    classifier_elapsed_ms = (time.perf_counter() - start_classifier) * 1000

    classes = [str(item) for item in (classifier_output.get("classes") or [])]
    raw_scores = classifier_output.get("scores") or []
    scores = [float(item) for item in raw_scores]
    scores_by_class = {
        label: float(score)
        for label, score in zip(classes, scores)
    }
    raw_probabilities = classifier_output.get("probabilities") or []
    probabilities = [float(item) for item in raw_probabilities]
    if len(probabilities) != len(classes) or not _scores_are_probability_like(probabilities):
        probabilities = _softmax_normalize_scores(scores)
    probabilities_by_class = {
        label: float(probability)
        for label, probability in zip(classes, probabilities)
    }
    class_description_rows = [
        {
            "class": label,
            "description": ECG_CLASS_DESCRIPTIONS.get(label, label),
        }
        for label in classes
    ]
    raw_score_rows = [
        {
            "class": label,
            "description": ECG_CLASS_DESCRIPTIONS.get(label, label),
            "score": float(score),
        }
        for label, score in zip(classes, scores)
    ]
    ui_prediction_score_rows = [
        {
            "class": label,
            "description": ECG_CLASS_DESCRIPTIONS.get(label, label),
            "score": float(probability),
        }
        for label, probability in zip(classes, probabilities)
    ]
    ecg_classifier_details = {
        "classifier_type": str(classifier_output.get("classifier_type") or ""),
        "checkpoint_path": str(classifier_output.get("checkpoint_path") or ""),
        "medsiglip_model_id": str(classifier_output.get("medsiglip_model_id") or ""),
        "device": str(classifier_output.get("device") or ""),
        "classes": classes,
        "scores": scores,
        "display_scores": probabilities,
        "scores_by_class": scores_by_class,
        "probabilities": probabilities,
        "probabilities_by_class": probabilities_by_class,
        "predictions": [int(item) for item in (classifier_output.get("predictions") or [])],
        "predictions_by_class": {
            str(label): int(value)
            for label, value in (classifier_output.get("predictions_by_class") or {}).items()
        },
        "predicted_labels": [
            str(item) for item in (classifier_output.get("predicted_labels") or [])
        ],
        "threshold": float(classifier_output.get("threshold", 0.5)),
        "gate_weights": [float(item) for item in (classifier_output.get("gate_weights") or [])],
        "num_experts": int(classifier_output.get("num_experts") or 0),
        "scoring_mode": str(classifier_output.get("scoring_mode") or ""),
    }
    logger.info(
        "[upload-analysis][ecg] classifier inference done id=%s classifier_type=%s threshold=%.3f predicted=%s top3=%s elapsed_ms=%.1f",
        request_id,
        str(classifier_output.get("classifier_type") or ""),
        float(classifier_output.get("threshold", 0.5)),
        [str(item) for item in (classifier_output.get("predicted_labels") or [])],
        _top_scores_for_log(scores_by_class, top_k=3),
        classifier_elapsed_ms,
    )

    prompt = f"""Analyze this uploaded ECG image using both:
1) The actual ECG image.
2) The classifier scores computed from MedSigLIP embedding.

Record metadata:
- record_type: {record_type}
- title: {safe_title}

ECG classifier output:
- classes (ordered): {json.dumps(classes, ensure_ascii=False)}
- class_descriptions: {json.dumps(class_description_rows, ensure_ascii=False)}
- raw_logits (same order): {json.dumps(scores, ensure_ascii=False)}
- scores_by_class: {json.dumps(scores_by_class, ensure_ascii=False)}
- probabilities (same order): {json.dumps(probabilities, ensure_ascii=False)}
- probabilities_by_class: {json.dumps(probabilities_by_class, ensure_ascii=False)}
- raw_score_rows: {json.dumps(raw_score_rows, ensure_ascii=False)}
- prediction_score_rows: {json.dumps(ui_prediction_score_rows, ensure_ascii=False)}
- predicted_labels: {json.dumps(classifier_output.get("predicted_labels") or [], ensure_ascii=False)}
- threshold: {float(classifier_output.get("threshold", 0.5))}

{UPLOAD_ANALYSIS_SECTIONED_FORMAT}

Rules:
- Use Vietnamese for user-facing fields.
- Use proper Vietnamese diacritics (tone marks); do not remove accents.
- Use the image as primary evidence and classifier scores as supporting evidence.
- Mention the highest-scoring ECG class in the summary or key findings.
- Do not claim a diagnosis with absolute certainty.
- Keep the section headers exactly as written above.
- Use `- ` list items only under Key findings, Recommended follow-up, and Limitations.
- Keep summary under 120 words.
- Do not repeat template placeholders such as `<finding 1>` or `low|medium|high`.
- Do not answer in English.
- Do not return JSON.
- Do not include markdown fences.
"""

    parsed: Optional[dict[str, Any]] = None
    raw = ""
    repaired: dict[str, Any]
    multimodal_enabled, multimodal_reason = _multimodal_route_capability(
        route_name="upload_analysis",
        model=analysis_model,
    )
    if not multimodal_enabled:
        logger.warning(
            "[upload-analysis][ecg] multimodal skipped id=%s provider=%s model=%s reason=%s; using classifier-only fallback",
            request_id,
            settings.llm_provider,
            analysis_model,
            multimodal_reason,
        )
        repaired = _build_ecg_classifier_fallback_analysis(
            prediction_scores=ui_prediction_score_rows,
            predicted_labels=[
                str(item) for item in (classifier_output.get("predicted_labels") or [])
            ],
            reason=(
                "Đã bỏ qua phân tích ảnh từ LLM vì tuyến model hiện tại không cho phép ảnh; "
                "phần diễn giải này được dựng từ điểm bộ phân loại."
            ),
        )
    else:
        logger.info("[upload-analysis][ecg] medgemma call start id=%s", request_id)
        start_llm = time.perf_counter()
        try:
            raw = await llm_client.generate(
                model=analysis_model,
                prompt=prompt,
                system=UPLOAD_ANALYSIS_SYSTEM,
                images=[image_base64],
                stream=False,
                num_predict=768,
            )
            llm_elapsed_ms = (time.perf_counter() - start_llm) * 1000
            logger.info(
                "[upload-analysis][ecg] medgemma call done id=%s response_len=%s elapsed_ms=%.1f",
                request_id,
                len(raw or ""),
                llm_elapsed_ms,
            )
            parsed = _extract_json_object(raw)
            repaired = _repair_upload_analysis_payload(parsed, raw or "")
        except Exception as exc:
            llm_elapsed_ms = (time.perf_counter() - start_llm) * 1000
            diagnostic = _classify_llm_error(str(exc) or repr(exc))
            logger.exception(
                "[upload-analysis][ecg] medgemma call failed id=%s error=%s elapsed_ms=%.1f; using classifier-only fallback",
                request_id,
                diagnostic,
                llm_elapsed_ms,
            )
            repaired = _build_ecg_classifier_fallback_analysis(
                prediction_scores=ui_prediction_score_rows,
                predicted_labels=[
                    str(item) for item in (classifier_output.get("predicted_labels") or [])
                ],
                reason=(
                    "Mô hình phân tích ảnh ECG không phản hồi hợp lệ; "
                    "phần diễn giải này được dựng từ điểm bộ phân loại."
                ),
            )
    if _looks_like_low_quality_ecg_output(repaired, raw or ""):
        logger.warning(
            "[upload-analysis][ecg] low-quality llm output id=%s preview=%s",
            request_id,
            _preview_model_output_for_log(raw or ""),
        )
        repaired = _build_ecg_classifier_fallback_analysis(
            prediction_scores=ui_prediction_score_rows,
            predicted_labels=[
                str(item) for item in (classifier_output.get("predicted_labels") or [])
            ],
            reason="LLM trả về nội dung quá chung chung hoặc còn placeholder.",
        )

    if not _repaired_payload_has_meaningful_content(repaired):
        logger.warning(
            "[upload-analysis][ecg] invalid structured output id=%s preview=%s",
            request_id,
            _preview_model_output_for_log(raw or ""),
        )
        return _validated_upload_analysis_result(
            {
                **base_result,
                "status": "error",
                "summary": "Không thể tạo AI analysis.",
                "key_findings": [],
                "recommended_follow_up": [],
                "limitations": ["LLM returned invalid structured output."],
                "prediction_scores": ui_prediction_score_rows,
                "ecg_classifier": ecg_classifier_details,
            },
            fallback_message="Không thể tạo AI analysis.",
        )

    if not parsed:
        logger.info(
            "[upload-analysis][ecg] repaired non-json output id=%s preview=%s",
            request_id,
            _preview_model_output_for_log(raw or ""),
        )

    summary = _resolve_upload_analysis_summary(
        repaired,
        raw or "",
        fallback_message="Không thể tạo AI analysis.",
    )

    urgency = _normalize_upload_analysis_level(repaired.get("urgency"))
    confidence = _normalize_upload_analysis_level(repaired.get("confidence"))
    total_elapsed_ms = (time.perf_counter() - start_total) * 1000
    logger.info(
        "[upload-analysis][ecg] completed id=%s urgency=%s confidence=%s findings=%s follow_up=%s elapsed_ms=%.1f",
        request_id,
        urgency,
        confidence,
        len(repaired.get("key_findings") or []),
        len(repaired.get("recommended_follow_up") or []),
        total_elapsed_ms,
    )

    return _validated_upload_analysis_result({
        **base_result,
        "status": "completed",
        "summary": summary,
        "key_findings": repaired.get("key_findings") or [],
        "clinical_significance": repaired.get("clinical_significance"),
        "recommended_follow_up": repaired.get("recommended_follow_up") or [],
        "urgency": urgency,
        "confidence": confidence,
        "limitations": repaired.get("limitations") or [],
        # Persist UI-friendly probabilities while keeping raw classifier scores below for diagnostics.
        "prediction_scores": ui_prediction_score_rows,
        "ecg_classifier": ecg_classifier_details,
    }, fallback_message="Không thể tạo AI analysis.")


async def analyze_uploaded_record(
    *,
    record_type: str,
    title: Optional[str],
    extracted_text: Optional[str],
    image_base64: Optional[str] = None
) -> dict[str, Any]:
    """
    Analyze an uploaded medical record using configured upload-analysis model routing.

    Returns:
        JSON-serializable dict to store in medical_records.analysis_result
    """
    start_total = time.perf_counter()
    safe_title = (title or "Untitled record").strip()
    extracted = (extracted_text or "").strip()
    if len(extracted) > 12000:
        extracted = extracted[:12000]
    if (record_type or "").strip().lower() == "ecg" and image_base64:
        # ECG path is image-first; do not include OCR text in model prompt.
        extracted = ""

    cache_key = _build_upload_analysis_cache_key(
        record_type=record_type,
        title=safe_title,
        extracted_text=extracted,
        image_base64=image_base64,
    )
    cached_result = await _get_cached_upload_analysis(cache_key)
    if cached_result:
        logger.info(
            "[upload-analysis] cache hit record_type=%s title=%s text_len=%s has_image=%s",
            record_type,
            safe_title[:120],
            len(extracted),
            bool(image_base64),
        )
        return cached_result

    request_id = uuid.uuid4().hex[:8]

    timestamp = datetime.now(timezone.utc).isoformat()
    base_result = {
        "model": _resolve_upload_analysis_model(has_image=bool(image_base64)),
        "record_type": record_type,
        "generated_at": timestamp,
        "request_id": request_id,
    }
    analysis_model = str(base_result["model"])

    logger.info(
        "[upload-analysis] start id=%s provider=%s model=%s record_type=%s title=%s text_len=%s has_image=%s",
        request_id,
        settings.llm_provider,
        analysis_model,
        record_type,
        safe_title[:120],
        len(extracted),
        bool(image_base64),
    )

    ecg_fallback_reason: Optional[str] = None
    if (record_type or "").strip().lower() == "ecg" and image_base64:
        logger.info("[upload-analysis] routing ECG request to classifier flow id=%s", request_id)
        try:
            result = await _analyze_ecg_with_classifier(
                request_id=request_id,
                record_type=record_type,
                safe_title=safe_title,
                image_base64=image_base64,
                analysis_model=analysis_model,
                base_result=base_result,
            )
            await _store_upload_analysis_cache(cache_key, result)
            return result
        except Exception as exc:
            ecg_fallback_reason = _sanitize_text(str(exc) or repr(exc), max_len=500)
            logger.warning(
                "[upload-analysis] ECG classifier flow failed; using default flow id=%s error=%s",
                request_id,
                ecg_fallback_reason,
            )
            logger.exception(
                "[upload-analysis][ecg] classifier workflow failed id=%s error=%s; falling back to default flow",
                request_id,
                ecg_fallback_reason,
            )

    if not extracted and not image_base64:
        logger.warning("[upload-analysis] skipped id=%s: no OCR text and no image payload", request_id)
        result = {
            **base_result,
            "status": "skipped",
            "summary": "No extractable content found for AI analysis.",
            "key_findings": [],
            "recommended_follow_up": [],
            "limitations": ["No OCR text or image content was available."],
        }
        result = _validated_upload_analysis_result(
            result,
            fallback_message="No extractable content found for AI analysis.",
        )
        await _store_upload_analysis_cache(cache_key, result)
        return result

    prompt = f"""Analyze the uploaded medical record and return concise clinical insights.

Record metadata:
- record_type: {record_type}
- title: {safe_title}

Extracted text (OCR):
{extracted or "No OCR text available."}

{UPLOAD_ANALYSIS_SECTIONED_FORMAT}

Rules:
- Use Vietnamese for user-facing fields.
- Use proper Vietnamese diacritics (tone marks); do not remove accents.
- Keep the section headers exactly as written above.
- Use `- ` list items only under Key findings, Recommended follow-up, and Limitations.
- Keep summary under 120 words.
- Keep key_findings and recommended_follow_up concise and actionable.
- If data is limited, state that clearly in limitations.
- Do not return JSON.
- Do not include markdown fences.
"""

    model_available = await llm_client.check_model_available(analysis_model)
    if not model_available:
        logger.error(
            "[upload-analysis] model unavailable id=%s provider=%s model=%s",
            request_id,
            settings.llm_provider,
            analysis_model
        )
        return _validated_upload_analysis_result({
            **base_result,
            "status": "error",
            "summary": "AI analysis model is unavailable on this server.",
            "key_findings": [],
            "recommended_follow_up": [],
            "limitations": [f"Model not available: {analysis_model}"],
        }, fallback_message="AI analysis model is unavailable on this server.")

    images = [image_base64] if image_base64 else None

    try:
        raw = ""
        multimodal_error: Optional[str] = None

        multimodal_enabled = False
        if images:
            multimodal_enabled, multimodal_reason = _multimodal_route_capability(
                route_name="upload_analysis",
                model=analysis_model,
            )

        if images and multimodal_enabled:
            try:
                raw = await llm_client.generate(
                    model=analysis_model,
                    prompt=prompt,
                    system=UPLOAD_ANALYSIS_SYSTEM,
                    images=images,
                    stream=False,
                    num_predict=768,
                )
                logger.info(
                    "[upload-analysis] multimodal ok id=%s response_len=%s",
                    request_id,
                    len(raw or ""),
                )
            except Exception as exc:
                multimodal_error = _sanitize_text(str(exc) or repr(exc), max_len=500)
                logger.exception(
                    "[upload-analysis] multimodal failed id=%s provider=%s model=%s image_len=%s; falling back to text-only",
                    request_id,
                    settings.llm_provider,
                    analysis_model,
                    len(image_base64 or ""),
                )
        elif images:
            multimodal_error = _sanitize_text(multimodal_reason, max_len=500)
            logger.warning(
                "[upload-analysis] multimodal skipped id=%s provider=%s model=%s reason=%s",
                request_id,
                settings.llm_provider,
                analysis_model,
                multimodal_error,
            )
        else:
            raw = await llm_client.generate(
                model=analysis_model,
                prompt=prompt,
                system=UPLOAD_ANALYSIS_SYSTEM,
                images=None,
                stream=False,
                num_predict=768,
            )
            logger.info(
                "[upload-analysis] text-only primary ok id=%s response_len=%s",
                request_id,
                len(raw or ""),
            )

        # Retry text-only if multimodal call returned empty/garbled payload.
        should_retry_text_only = bool(extracted) or not images
        if (not raw or len(raw.strip()) < 10) and should_retry_text_only:
            logger.warning(
                "[upload-analysis] retrying text-only id=%s reason=%s",
                request_id,
                "multimodal_error" if multimodal_error else "short_or_empty_response",
            )
            raw = await llm_client.generate(
                model=analysis_model,
                prompt=prompt,
                system=UPLOAD_ANALYSIS_SYSTEM,
                images=None,
                stream=False,
                num_predict=768,
            )
            logger.info(
                "[upload-analysis] text-only retry ok id=%s response_len=%s",
                request_id,
                len(raw or ""),
            )

        if images and not raw and not extracted:
            limitations = []
            if ecg_fallback_reason:
                limitations.append("ECG classifier path failed, used default upload analysis flow.")
            if multimodal_error:
                limitations.append(f"Image analysis unavailable: {multimodal_error}")
            limitations.append("No OCR text was available for text-only fallback.")
            return _validated_upload_analysis_result(
                {
                    **base_result,
                    "status": "error",
                    "summary": "Không thể tạo AI analysis cho ảnh này với cấu hình hiện tại.",
                    "key_findings": [],
                    "recommended_follow_up": [],
                    "limitations": _sanitize_list(limitations, max_items=6, max_item_len=500),
                },
                fallback_message="Không thể tạo AI analysis cho ảnh này với cấu hình hiện tại.",
            )

        parsed = _extract_json_object(raw)
        repaired = _repair_upload_analysis_payload(parsed, raw or "")
        if not _repaired_payload_has_meaningful_content(repaired):
            limitations = ["LLM returned invalid structured output."]
            if ecg_fallback_reason:
                limitations.append("ECG classifier path failed, used default upload analysis flow.")
            if multimodal_error:
                limitations.append(f"Image analysis fallback was used: {multimodal_error}")

            logger.warning(
                "[upload-analysis] invalid structured output id=%s preview=%s",
                request_id,
                _preview_model_output_for_log(raw or ""),
            )
            return _validated_upload_analysis_result(
                {
                    **base_result,
                    "status": "error",
                    "summary": "Không thể tạo AI analysis.",
                    "key_findings": [],
                    "recommended_follow_up": [],
                    "limitations": _sanitize_list(limitations, max_items=6, max_item_len=500),
                },
                fallback_message="Không thể tạo AI analysis.",
            )

        if not parsed:
            logger.info(
                "[upload-analysis] repaired non-json output id=%s preview=%s",
                request_id,
                _preview_model_output_for_log(raw or ""),
            )

        summary = _resolve_upload_analysis_summary(
            repaired,
            raw or "",
            fallback_message="Không thể tạo AI analysis.",
        )

        urgency = _normalize_upload_analysis_level(repaired.get("urgency"))
        confidence = _normalize_upload_analysis_level(repaired.get("confidence"))

        result: dict[str, Any] = {
            **base_result,
            "status": "completed",
            "summary": summary,
            "key_findings": repaired.get("key_findings") or [],
            "clinical_significance": repaired.get("clinical_significance"),
            "recommended_follow_up": repaired.get("recommended_follow_up") or [],
            "urgency": urgency,
            "confidence": confidence,
            "limitations": repaired.get("limitations") or [],
        }
        if ecg_fallback_reason:
            result["limitations"] = _sanitize_list(
                (result.get("limitations") or [])
                + ["ECG classifier path failed, used default upload analysis flow."],
                max_items=6,
                max_item_len=500,
            )
        if multimodal_error:
            result["limitations"] = _sanitize_list(
                (result.get("limitations") or []) + [
                    f"Image analysis fallback was used: {multimodal_error}",
                ],
                max_items=6,
                max_item_len=500,
            )
        logger.info(
            "[upload-analysis] completed id=%s urgency=%s confidence=%s findings=%s follow_up=%s used_fallback=%s elapsed_ms=%.1f",
            request_id,
            result.get("urgency"),
            result.get("confidence"),
            len(result.get("key_findings") or []),
            len(result.get("recommended_follow_up") or []),
            bool(ecg_fallback_reason),
            (time.perf_counter() - start_total) * 1000,
        )
        result = _validated_upload_analysis_result(
            result,
            fallback_message="Không thể tạo AI analysis.",
        )
        await _store_upload_analysis_cache(cache_key, result)
        return result
    except Exception as exc:
        detail = _sanitize_text(str(exc) or repr(exc), max_len=500)
        reason = _classify_llm_error(detail)
        logger.exception(
            "[upload-analysis] failed id=%s provider=%s model=%s reason=%s error=%s has_image=%s text_len=%s title=%s elapsed_ms=%.1f",
            request_id,
            settings.llm_provider,
            analysis_model,
            reason,
            detail,
            bool(image_base64),
            len(extracted),
            safe_title[:120],
            (time.perf_counter() - start_total) * 1000,
        )
        return _validated_upload_analysis_result({
            **base_result,
            "status": "error",
            "summary": "AI analysis is temporarily unavailable for this file.",
            "key_findings": [],
            "recommended_follow_up": [],
            "limitations": ["Could not complete AI analysis at this time."],
        }, fallback_message="AI analysis is temporarily unavailable for this file.")
    finally:
        # Keep memory usage stable after one-shot upload analysis.
        await llm_client.unload(analysis_model)


async def translate_vi_to_en(text: str) -> str:
    """
    Backward-compatible passthrough.
    """
    return text


async def translate_en_to_vi(text: str) -> str:
    """
    Backward-compatible passthrough.
    """
    return text


async def medical_reasoning(
    query_en: str,
    patient_context: str,
    image_base64: Optional[str] = None
) -> str:
    """
    Generate medical response using MedGemma with RAG context.
    
    Args:
        query_en: Query in English
        patient_context: Aggregated patient context from RAG
        image_base64: Optional base64-encoded medical image
        
    Returns:
        Medical response in English
    """
    prompt = f"""## Patient Context
{patient_context}

## User Query
{query_en}

Please provide a helpful, accurate medical response based on the patient's context."""

    images = [image_base64] if image_base64 else None
    prompt_suffix = ""
    if images:
        multimodal_enabled, multimodal_reason = _multimodal_route_capability(
            route_name="medical_reasoning",
            model=settings.medical_model,
        )
        if not multimodal_enabled:
            logger.warning(
                "[medical-reasoning] multimodal skipped provider=%s model=%s reason=%s",
                settings.llm_provider,
                settings.medical_model,
                multimodal_reason,
            )
            images = None
            prompt_suffix = (
                "\n\nNote: Image analysis was skipped because the configured LLM route "
                "is not enabled for images."
            )

    response = await llm_client.generate(
        model=settings.medical_model,
        prompt=f"{prompt}{prompt_suffix}",
        system=MEDICAL_REASONING_SYSTEM,
        images=images,
        stream=False
    )
    return response


async def process_medical_query(
    user_input_vi: str,
    patient_id: UUID,
    image_path: Optional[str] = None
) -> AsyncGenerator[dict, None]:
    """Patient query pipeline with streaming updates."""
    start_total = time.perf_counter()
    image_base64 = None
    
    # Load image if provided
    if image_path:
        path = Path(image_path)
        if path.exists():
            with open(path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    # ========== STEP A: Analyze Input ==========
    yield {
        "stage": "verifying_input",
        "message": "Đang phân tích câu hỏi...",
        "progress": 0.1
    }
    start_step_a = time.perf_counter()
    query_en = user_input_vi
    elapsed_a = (time.perf_counter() - start_step_a) * 1000
    logger.info(f"[LLM] step_a_translate_input: Took {elapsed_a:.1f} ms")
    
    yield {
        "stage": "verified_input",
        "message": "Đã hiểu câu hỏi",
        "progress": 0.25,
    }

    # ========== STEP B: Medical Reasoning with RAG ==========
    yield {
        "stage": "retrieving_context",
        "message": "Đang tìm kiếm hồ sơ y tế liên quan...",
        "progress": 0.35
    }
    start_step_b_context = time.perf_counter()
    # Get patient context via RAG
    # Use original Vietnamese query for retrieval (records are primarily Vietnamese)
    patient_context = await get_patient_context(
        patient_id=patient_id,
        query=user_input_vi,
        max_chunks=10
    )
    elapsed_b_context = (time.perf_counter() - start_step_b_context) * 1000
    logger.info(f"[LLM] step_b_context: Took {elapsed_b_context:.1f} ms")
    
    yield {
        "stage": "medical_reasoning",
        "message": "Đang phân tích y khoa...",
        "progress": 0.5
    }
    
    # Medical reasoning with MedGemma
    start_step_b_reasoning = time.perf_counter()
    response_en = await medical_reasoning(
        query_en=query_en,
        patient_context=patient_context,
        image_base64=image_base64
    )
    elapsed_b_reasoning = (time.perf_counter() - start_step_b_reasoning) * 1000
    logger.info(f"[LLM] step_b_medical_reasoning: Took {elapsed_b_reasoning:.1f} ms")
    
    yield {
        "stage": "medical_reasoning",
        "message": "Hoàn thành phân tích",
        "progress": 0.7,
    }
    
    # ========== Memory Optimization: Unload MedGemma ==========
    await llm_client.unload(settings.medical_model)
    
    # ========== STEP C: Finalize Output ==========
    yield {
        "stage": "formatting_output",
        "message": "Đang chuẩn bị phản hồi...",
        "progress": 0.85
    }
    start_step_c = time.perf_counter()
    response_vi = response_en
    elapsed_c = (time.perf_counter() - start_step_c) * 1000
    logger.info(f"[LLM] step_c_translate_output: Took {elapsed_c:.1f} ms")
    
    yield {
        "stage": "complete",
        "message": "Hoàn thành",
        "progress": 1.0,
        "response": response_vi,
    }
    elapsed_total = (time.perf_counter() - start_total) * 1000
    logger.info(f"[LLM] pipeline_total: Took {elapsed_total:.1f} ms")


async def generate_patient_profile_summary(
    patient_id: UUID,
) -> dict[str, Any]:
    """
    Generate an AI clinical summary for a patient profile.

    Uses MedGemma to synthesize patient data into a structured clinical
    overview following the Problem List / POMR medical format.

    Args:
        patient_id: Patient UUID

    Returns:
        Dict with summary text, model name, and generation timestamp.
    """
    start_total = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]
    logger.info(
        "[patient-summary] start id=%s patient=%s model=%s",
        request_id,
        str(patient_id),
        settings.medical_model,
    )

    # Gather patient context via RAG (includes demographics, conditions,
    # medications, vitals, appointments, medical records).
    patient_context = await get_patient_context(patient_id)

    summary_prompt = f"""Dựa trên thông tin bệnh nhân dưới đây, hãy tạo một bản tóm tắt lâm sàng ngắn gọn theo chuẩn y khoa.

{patient_context}

Hãy viết bản tóm tắt theo đúng định dạng sau (bằng tiếng Việt, có dấu):

## Danh sách vấn đề (Problem List)
Liệt kê các bệnh lý/vấn đề sức khỏe hiện tại, kèm mã ICD-10 nếu có.
Ví dụ: 1. [E11] Đái tháo đường type 2 — Đang điều trị

## Thuốc đang dùng (Current Medications)
Liệt kê tên thuốc, liều lượng, tần suất dùng.

## Dị ứng (Allergies)
Liệt kê các dị ứng đã biết hoặc ghi "Chưa ghi nhận dị ứng" nếu không có.

## Diễn tiến bệnh (Disease Progress)
Mô tả ngắn gọn diễn tiến của các bệnh lý chính dựa trên dữ liệu sinh hiệu và lịch sử khám bệnh. Ví dụ: xu hướng đường huyết, huyết áp qua các lần đo gần đây, tuân thủ điều trị.

## Tóm tắt sinh hiệu gần nhất (Recent Vitals)
Tóm tắt các chỉ số sinh hiệu gần nhất trên một dòng.
Ví dụ: HA: 120/80 mmHg | Nhịp tim: 72 bpm | SpO₂: 98% | Đường huyết: 5.6 mmol/L

## Đánh giá lâm sàng (Clinical Assessment)
Viết 2-3 câu đánh giá tổng quát tình trạng sức khỏe của bệnh nhân, bao gồm mức độ kiểm soát bệnh và các khuyến nghị theo dõi.

QUY TẮC:
- Viết bằng tiếng Việt có dấu.
- Không dùng placeholder như [Insert...], [TODO], [N/A].
- Nếu thiếu dữ liệu, ghi rõ "Chưa có dữ liệu" thay vì bỏ trống.
- Giữ ngắn gọn, súc tích, chuyên nghiệp.
- Đây là tóm tắt cho bác sĩ xem trên hồ sơ bệnh nhân."""

    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        raw = await llm_client.generate(
            model=settings.medical_model,
            prompt=summary_prompt,
            system="Bạn là trợ lý AI y khoa chuyên tạo tóm tắt lâm sàng cho bác sĩ. Viết ngắn gọn, chuyên nghiệp, bằng tiếng Việt có dấu.",
            stream=False,
            num_predict=1200,
        )
        summary_text = (raw or "").strip()
        if not summary_text:
            summary_text = "Không thể tạo tóm tắt lâm sàng. Vui lòng thử lại sau."

        # Post-process malformed markdown from model output so UI renders
        # section headers/lists consistently (similar readability as chat responses).
        summary_text = _normalize_patient_summary_markdown(summary_text)

        elapsed_ms = (time.perf_counter() - start_total) * 1000
        logger.info(
            "[patient-summary] completed id=%s patient=%s response_len=%s elapsed_ms=%.1f",
            request_id,
            str(patient_id),
            len(summary_text),
            elapsed_ms,
        )

        return {
            "summary": summary_text,
            "generated_at": timestamp,
            "model": settings.medical_model,
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_total) * 1000
        logger.exception(
            "[patient-summary] failed id=%s patient=%s error=%s elapsed_ms=%.1f",
            request_id,
            str(patient_id),
            str(exc)[:300],
            elapsed_ms,
        )
        return {
            "summary": "Tạo tóm tắt lâm sàng thất bại. Vui lòng thử lại sau.",
            "generated_at": timestamp,
            "model": settings.medical_model,
            "error": str(exc)[:300],
        }
    finally:
        await llm_client.unload(settings.medical_model)


async def generate_clinical_summary(
    consultation_id: UUID,
    patient_id: UUID
) -> str:
    """
    Generate clinical notes summary from consultation history.
    
    Args:
        consultation_id: Consultation UUID
        patient_id: Patient UUID
        
    Returns:
        Clinical summary in Vietnamese
    """
    from app.db.database import get_supabase
    
    supabase = get_supabase()
    
    # Get consultation messages
    consultation = supabase.table("consultations").select(
        "messages, chief_complaint"
    ).eq("id", str(consultation_id)).single().execute()
    
    if not consultation.data:
        return "Không tìm thấy thông tin tư vấn."
    
    # Get patient context
    patient_context = await get_patient_context(patient_id)
    
    # Format messages for summary
    messages = consultation.data.get("messages", [])
    chief_complaint = consultation.data.get("chief_complaint", "N/A")
    
    messages_text = "\n".join([
        f"- {m.get('role', 'user')}: {m.get('content', '')}"
        for m in messages
    ])
    
    summary_prompt = f"""Based on the following consultation, generate a structured clinical summary for medical records.

## Patient Context
{patient_context}

## Chief Complaint
{chief_complaint}

## Consultation Messages
{messages_text}

Return the summary in markdown using these top-level sections:

1. Medical History
- Chronic Conditions: include only if mentioned in consultation/context; add timeline/date when available.
- Past Surgeries: include only if mentioned; list each surgery with date and outcomes when available.
- Hospitalizations: include only if mentioned; include reason and timing.
- Medications History: include only if mentioned; include past meds and discontinuation reasons when available.
- Allergies: include only if explicitly mentioned; include trigger and reaction details.
- Psychiatric History: include only if explicitly mentioned; include diagnosis/treatment details when available.

2. Family Medical History
- Family History of Chronic Conditions: include only if explicitly mentioned.
- Family History of Mental Health Conditions: include only if explicitly mentioned.
- Family History of Genetic Conditions: include only if explicitly mentioned.

3. Immunization Records
- Vaccines Administered: include only if explicitly mentioned, with date if available.
- Vaccines Due: include only if explicitly mentioned.

4. Treatment History
- Previous Treatments: include only if mentioned; include outcomes when available.
- Physiotherapy: include only if applicable and mentioned.
- Other Relevant Treatments: include only if relevant and mentioned.

5. Treatment Records
- Regular Checkup Entries (vital signs as part of treatment records):
  - Date of examination
  - Reason for visit
  - Doctor comments on test results
  - Patient progress
  - Treatment plan
  - Doctor notes
- Medical Records (test results only): include only lab/xray/ecg/ct/mri.
  - Medical Images format:
    0) Medical file attached (if available)
    1) Doctor's test result description
    2) Doctor's final conclusion
    3) AI analysis
  - Lab Results format:
    Include a markdown table with columns: Sample Information | Test Name | Numerical Result | Unit | Flag/Status | Doctor's Notes (optional)

Rules:
- Do not use placeholders like [Enter ...].
- Do not fabricate details; include only information present in context/messages.
- Omit bullets/subsections that are not mentioned instead of writing "N/A".
- Keep wording concise and clinically clear.
"""

    # Generate summary with MedGemma
    summary_en = await llm_client.generate(
        model=settings.medical_model,
        prompt=summary_prompt,
        system="You are a medical documentation specialist. Generate professional clinical notes.",
        stream=False
    )
    
    await llm_client.unload(settings.medical_model)
    
    return summary_en


async def check_system_health() -> dict:
    """
    Check if all required models are available.
    
    Returns:
        Health status dict
    """
    llm_ok = await llm_client.health_check()
    provider = (settings.llm_provider or "vertex").lower()
    upload_analysis_model = _resolve_upload_analysis_model(has_image=True)
    upload_images_enabled, upload_images_reason = _multimodal_route_capability(
        route_name="upload_analysis",
        model=upload_analysis_model,
    )
    medical_reasoning_images_enabled, medical_reasoning_images_reason = _multimodal_route_capability(
        route_name="medical_reasoning",
        model=settings.medical_model,
    )

    if not llm_ok:
        return {
            "status": "unhealthy",
            "provider": provider,
            "llm": False,
            "message": f"{provider} provider is not reachable",
            "capabilities": {
                "upload_analysis_images": {
                    "enabled": upload_images_enabled,
                    "model": upload_analysis_model,
                    "reason": upload_images_reason,
                },
                "medical_reasoning_images": {
                    "enabled": medical_reasoning_images_enabled,
                    "model": settings.medical_model,
                    "reason": medical_reasoning_images_reason,
                },
            },
        }
    
    models_status = {
        "medical_model": await llm_client.check_model_available(
            settings.medical_model
        ),
        "upload_analysis_model": await llm_client.check_model_available(
            upload_analysis_model
        ),
        "embedding_model": await llm_client.check_model_available(
            settings.embedding_model
        )
    }
    
    all_available = all(models_status.values())
    
    return {
        "status": "healthy" if all_available else "degraded",
        "provider": provider,
        "llm": True,
        "models": models_status,
        "capabilities": {
            "upload_analysis_images": {
                "enabled": upload_images_enabled,
                "model": upload_analysis_model,
                "reason": upload_images_reason,
            },
            "medical_reasoning_images": {
                "enabled": medical_reasoning_images_enabled,
                "model": settings.medical_model,
                "reason": medical_reasoning_images_reason,
            },
        },
        "message": "All systems operational" if all_available else "Some models missing"
    }
