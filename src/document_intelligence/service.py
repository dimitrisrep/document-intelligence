"""Trusted model loading and exports shared by all interfaces."""

import csv
import io
from pathlib import Path

import joblib
import sklearn

from document_intelligence.contracts import FIELDS
from document_intelligence.extraction import extract_fields
from document_intelligence.reading import read_document


class DocumentExtractor:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.model = None
        path = self.directory / "extractor.joblib"
        if path.exists():
            bundle = joblib.load(path)  # Never load an artifact supplied in an upload.
            if (
                bundle.get("schema_version") != 1
                or bundle.get("sklearn_version") != sklearn.__version__
            ):
                raise ValueError("Extractor schema or sklearn version mismatch.")
            self.model = bundle["model"]

    def extract(self, content: bytes):
        if self.model is None:
            raise RuntimeError("No trained extraction artifact.")
        return extract_fields(read_document(content), self.model)


def csv_export(result):
    def safe(value):
        value = str(value)
        if value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
            return "'" + value
        return value

    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["field", "value", "model_score", "status"])
    for field in FIELDS:
        info = result["fields"].get(field)
        writer.writerow(
            [
                field,
                safe(info["value"]) if info else "",
                info["model_score"] if info else "",
                result["status"],
            ]
        )
    return stream.getvalue()
