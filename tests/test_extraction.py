import csv
import io

import numpy as np
import pytest

from document_intelligence.contracts import Line
from document_intelligence.extraction import extract_fields
from document_intelligence.layout import assemble_rows
from document_intelligence.normalization import money, receipt_date
from document_intelligence.service import csv_export


class ControlledModel:
    classes_ = np.array(["address", "company", "date", "other", "total"])

    def predict_proba(self, records):
        result = []
        for row in records:
            field = (
                "company"
                if row["text"].startswith("ACME")
                else "date"
                if "Date" in row["text"]
                else "total"
                if row["text"].startswith("Total")
                else "other"
            )
            result.append([0.96 if c == field else 0.01 for c in self.classes_])
        return np.asarray(result)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("RM 1,234.56", "1234.56"),
        ("1.234,56", "1234.56"),
        ("0.00", "0.00"),
        ("-12,30", "-12.30"),
        ("invalid", None),
        ("1.234", None),
        ("NaN", None),
    ],
)
def test_money_is_exact_and_rejects_ambiguous_separator(raw, expected):
    assert money(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Date: 29/02/2024", "2024-02-29"),
        ("2025-12-01", "2025-12-01"),
        ("31/02/2024", None),
    ],
)
def test_dates_are_validated(raw, expected):
    assert receipt_date(raw) == expected


def test_evidence_arithmetic_and_missing_fields_are_visible():
    texts = [
        "ACME SDN BHD",
        "Date: 01/06/2024",
        "Subtotal 10.00",
        "Tax 2.00",
        "Total RM 12.00",
    ]
    lines = [
        Line(t, (10, 10 + i * 30, 300, 30 + i * 30), 400, 400)
        for i, t in enumerate(texts)
    ]
    result = extract_fields(lines, ControlledModel())
    assert result["fields"]["total"]["value"] == "12.00"
    assert result["fields"]["total"]["evidence"][0]["quote"] == texts[-1]
    assert result["validation"]["arithmetic"] == "matched"
    assert "missing_address" in result["validation"]["issues"]
    assert result["status"] == "review_required"
    assert result["automatic_financial_actions"] == []
    lines[-1] = Line("Total RM 13.00", (10, 130, 300, 150), 400, 400)
    assert (
        extract_fields(lines, ControlledModel())["validation"]["arithmetic"]
        == "mismatch"
    )


def test_layout_joins_label_to_value_without_crossing_pages_or_rows():
    lines = [
        Line("12.00", (250, 100, 310, 120), 400, 500),
        Line("TOTAL", (10, 101, 80, 121), 400, 500),
        Line("CASH", (10, 130, 80, 150), 400, 500),
        Line("PAGE TWO", (10, 100, 90, 120), 400, 500, page=2),
    ]
    result = assemble_rows(lines)
    assert [r.text for r in result] == ["TOTAL 12.00", "CASH", "PAGE TWO"]
    assert result[0].bbox == (10, 100, 310, 121)


def test_csv_does_not_execute_spreadsheet_formulas():
    result = {
        "status": "review_required",
        "fields": {"company": {"value": " =SUM(1,2)", "model_score": 0.8}},
    }
    rows = list(csv.reader(io.StringIO(csv_export(result))))
    assert rows[1][1].startswith("'")
