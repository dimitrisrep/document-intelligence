from document_intelligence.training import field_metrics, training_label


def test_weak_total_label_survives_row_assembly():
    entities = {
        "company": "ACME SDN BHD",
        "address": "10 EXAMPLE ROAD",
        "date": "01/06/2024",
        "total": "12.00",
    }
    assert training_label("TOTAL RM 12.00", entities) == "total"
    assert training_label("Subtotal RM 10.00", entities) == "other"
    assert training_label("Date: 01/06/2024 12:30", entities) == "date"


def test_missing_predictions_count_as_errors():
    record = {
        "key": "example",
        "entities": {
            "company": "ACME",
            "address": "10 ROAD",
            "date": "01/06/2024",
            "total": "12.00",
        },
    }
    result = field_metrics([record], [{"fields": {}}])
    assert result["mean_field_exact_match"] == 0
    assert len(result["failures"]) == 4
