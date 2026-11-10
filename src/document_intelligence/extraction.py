"""Field values are selected from observed lines, never generated."""

import re
from decimal import Decimal

from document_intelligence.contracts import FIELDS
from document_intelligence.normalization import amounts, money, receipt_date


def extract_fields(lines, model):
    if not lines:
        return {
            "status": "review_required",
            "fields": {f: None for f in FIELDS},
            "validation": {"issues": ["no_readable_text"], "arithmetic": "not_checked"},
            "line_count": 0,
        }
    records = [line.to_record(i, len(lines)) for i, line in enumerate(lines)]
    probabilities = model.predict_proba(records)
    classes = list(model.classes_)
    fields = {}
    for field in FIELDS:
        if field not in classes:
            fields[field] = None
            continue
        scores = probabilities[:, classes.index(field)]
        candidates = []
        for i, line in enumerate(lines):
            value = line.text.strip()
            if field == "date":
                value = receipt_date(value)
            elif field == "total":
                found = amounts(value)
                value = money(found[-1]) if found else None
            if value and scores[i] >= 0.30:
                candidates.append((i, value, float(scores[i])))
        if not candidates:
            fields[field] = None
            continue
        if field == "address":
            selected = [
                row
                for row in candidates
                if probabilities[row[0]].argmax() == classes.index(field)
            ]
            selected = selected or [max(candidates, key=lambda r: r[2])]
            selected = sorted(
                selected,
                key=lambda r: (
                    lines[r[0]].page,
                    lines[r[0]].bbox[1],
                    lines[r[0]].bbox[0],
                ),
            )
            value = " ".join(row[1] for row in selected)
            score = min(row[2] for row in selected)
        else:
            selected = [max(candidates, key=lambda r: r[2])]
            _, value, score = selected[0]
        fields[field] = {
            "value": value,
            "model_score": score,
            "score_type": "uncalibrated_line_class_score",
            "evidence": [
                {
                    "quote": lines[i].text,
                    "bbox": list(lines[i].bbox),
                    "page": lines[i].page,
                    "source": lines[i].source,
                    "coordinate_units": "page_points"
                    if lines[i].source == "pdf_embedded_text"
                    else "image_pixels",
                    "page_size": {"width": lines[i].width, "height": lines[i].height},
                    "ocr_confidence": lines[i].confidence,
                }
                for i, _, _ in selected
            ],
        }
    supplements = {}
    for name, pattern in [
        ("subtotal", r"\bsub\s*total\b"),
        ("tax", r"\b(?:tax|gst|sst|vat)\b|φπα"),
    ]:
        for line in lines:
            if re.search(pattern, line.text, re.I) and (found := amounts(line.text)):
                supplements[name] = {
                    "value": money(found[-1]),
                    "method": "label_rule_unbenchmarked",
                    "evidence": line.text,
                }
    all_text = " ".join(line.text for line in lines)
    currency = (
        "MYR"
        if re.search(r"\b(?:RM|MYR)\b", all_text)
        else "EUR"
        if "€" in all_text
        else None
    )
    issues = [f"missing_{field}" for field, value in fields.items() if value is None]
    issues += [
        f"low_score_{field}"
        for field, value in fields.items()
        if value and value["model_score"] < 0.65
    ]
    arithmetic = "not_checked"
    if fields["total"] and supplements.get("subtotal") and supplements.get("tax"):
        total = Decimal(fields["total"]["value"])
        expected = Decimal(supplements["subtotal"]["value"]) + Decimal(
            supplements["tax"]["value"]
        )
        arithmetic = (
            "matched" if abs(total - expected) <= Decimal("0.02") else "mismatch"
        )
        if arithmetic == "mismatch":
            issues.append("subtotal_plus_tax_does_not_match_total")
    return {
        "status": "review_required",
        "fields": fields,
        "supplements": supplements,
        "currency": currency,
        "validation": {"issues": issues, "arithmetic": arithmetic},
        "line_count": len(lines),
        "automatic_financial_actions": [],
    }
