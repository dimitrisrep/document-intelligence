import hashlib
import json
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import GroupShuffleSplit

from document_intelligence.contracts import FIELDS
from document_intelligence.data import (
    CHECKSUMS,
    DATASET_ID,
    REVISION,
    annotation_lines,
    load_records,
)
from document_intelligence.extraction import extract_fields
from document_intelligence.features import make_model
from document_intelligence.normalization import (
    amounts,
    compact,
    money,
    normalized_field,
)


def training_label(text, entities):
    """Weak region alignment from gold field strings, not official region NER labels."""
    normalized = compact(text)
    if not normalized:
        return "other"
    candidates = []
    for field in FIELDS:
        target = compact(entities[field])
        if not target:
            continue
        score = SequenceMatcher(None, normalized, target).ratio()
        observed = normalized_field(field, text)
        if field == "total" and (found := amounts(text)):
            observed = money(found[-1])
        exact_value = observed is not None and observed == normalized_field(
            field, entities[field]
        )
        if field in {"company", "address"}:
            if len(normalized) >= 5 and normalized in target:
                score = max(score, 0.93)
            if score >= 0.84:
                candidates.append((score, field))
        elif exact_value:
            candidates.append((1.0, field))
    return max(candidates)[1] if candidates else "other"


def flatten(records):
    features, labels = [], []
    for record in records:
        lines = annotation_lines(record)
        for i, line in enumerate(lines):
            features.append(line.to_record(i, len(lines)))
            labels.append(training_label(line.text, record["entities"]))
    return features, labels


def field_metrics(records, predictions):
    matches = {field: 0 for field in FIELDS}
    predicted_count = {field: 0 for field in FIELDS}
    all_correct = 0
    failures = []
    for record, result in zip(records, predictions, strict=True):
        correct = 0
        for field in FIELDS:
            info = result["fields"].get(field)
            value = info["value"] if info else None
            if value is not None:
                predicted_count[field] += 1
            expected = normalized_field(field, record["entities"][field])
            same = (
                value is not None
                and expected is not None
                and normalized_field(field, value) == expected
            )
            matches[field] += int(same)
            correct += int(same)
            if not same:
                failures.append(
                    {
                        "key": record["key"],
                        "field": field,
                        "predicted": value,
                        "expected": record["entities"][field],
                    }
                )
        all_correct += correct == len(FIELDS)
    count = len(records)
    true_positive = sum(matches.values())
    precision = true_positive / max(sum(predicted_count.values()), 1)
    recall = true_positive / max(count * len(FIELDS), 1)
    return {
        "receipts": count,
        "field_exact_match": {
            field: matches[field] / max(count, 1) for field in FIELDS
        },
        "mean_field_exact_match": float(
            np.mean([matches[f] / max(count, 1) for f in FIELDS])
        ),
        "all_fields_exact_match": all_correct / max(count, 1),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "predicted_field_counts": predicted_count,
        "failures": failures,
    }


def train(directory: Path, output: Path):
    records = load_records(directory, "train")
    groups = np.array([compact(row["entities"]["company"]) for row in records])
    fit_ids, val_ids = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            records, groups=groups
        )
    )
    assert not set(groups[fit_ids]) & set(groups[val_ids])
    fit, validation = ([records[i] for i in indices] for indices in [fit_ids, val_ids])
    features, labels = flatten(fit)
    if not set(FIELDS) <= set(labels):
        raise ValueError("Training alignment did not identify all fields.")
    best = None
    comparisons = {}
    for c in [0.5, 2.0]:
        model = make_model(c).fit(features, labels)
        results = [extract_fields(annotation_lines(r), model) for r in validation]
        metric = field_metrics(validation, results)["mean_field_exact_match"]
        comparisons[str(c)] = {"validation_mean_field_exact_match": metric}
        if best is None or metric > best[0]:
            best = (metric, c, model)
    _, c, model = best
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": 1,
            "sklearn_version": sklearn.__version__,
            "model": model,
            "fields": FIELDS,
            "dataset_id": DATASET_ID,
            "revision": REVISION,
        },
        output / "extractor.joblib",
    )
    report = {
        "dataset": DATASET_ID,
        "revision": REVISION,
        "license": "CC BY 4.0",
        "checksums": CHECKSUMS,
        "seed": 42,
        "alignment": "weak region labels derived from field strings; not official NER labels",
        "fit_receipts": len(fit),
        "validation_receipts": len(validation),
        "fit_regions": len(features),
        "region_class_counts": {
            label: labels.count(label) for label in sorted(set(labels))
        },
        "split": "vendor-grouped development partition from official train; no official test used for selection",
        "fit_keys": [r["key"] for r in fit],
        "validation_keys": [r["key"] for r in validation],
        "vendor_group_overlap": 0,
        "selected_c": c,
        "candidates": comparisons,
        "sklearn_version": sklearn.__version__,
        "test_evaluation": "run evaluate separately after selection",
    }
    (output / "training.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def evaluate(directory: Path, output: Path, ocr_count=40):
    from document_intelligence.reading import read_document

    bundle = joblib.load(output / "extractor.joblib")
    model = bundle["model"]
    records = load_records(directory, "test")
    oracle_results = [extract_fields(annotation_lines(r), model) for r in records]
    report = {
        "dataset": DATASET_ID,
        "revision": REVISION,
        "official_test_receipts": len(records),
        "metric_definition": "case/punctuation-insensitive text match, ISO date equality and Decimal total equality",
        "oracle_transcription_kie": field_metrics(records, oracle_results),
        "ocr_sample_selection": "lowest SHA256(key) values, independent of labels",
        "ocr_failures": [],
    }
    if ocr_count:
        selected = sorted(
            records, key=lambda r: hashlib.sha256(r["key"].encode()).hexdigest()
        )[:ocr_count]
        chosen = {r["key"] for r in selected}
        images = {
            r["key"]: r
            for r in load_records(directory, "test", images=True)
            if r["key"] in chosen
        }
        predictions = []
        for i, row in enumerate(selected):
            try:
                lines = read_document(images[row["key"]]["image"]["bytes"])
                predictions.append(extract_fields(lines, model))
            except (RuntimeError, ValueError) as exc:
                predictions.append({"fields": {f: None for f in FIELDS}})
                report["ocr_failures"].append({"key": row["key"], "reason": str(exc)})
            print(
                json.dumps({"ocr_evaluated": i + 1, "total": len(selected)}), flush=True
            )
        report["tesseract_end_to_end"] = field_metrics(selected, predictions)
        report["ocr_sample_keys"] = [r["key"] for r in selected]
    report["limitations"] = [
        "Malaysian English receipts; no Greek invoice benchmark.",
        "Different track sizes: full annotated-text test versus a fixed OCR subset.",
        "Field metrics are project normalization metrics, not the official SROIE leaderboard script.",
        "Tax/subtotal rules are tested on fixtures only; no tax benchmark is claimed.",
    ]
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
