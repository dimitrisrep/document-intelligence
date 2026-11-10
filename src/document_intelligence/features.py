import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from document_intelligence.normalization import amounts, receipt_date


def line_texts(records):
    return [r["text"] for r in records]


def line_geometry(records):
    rows = []
    for r in records:
        text = r["text"]
        x0, y0, x1, y1 = r["bbox"]
        width, height = r["width"], r["height"]
        rows.append(
            [
                x0 / width,
                y0 / height,
                x1 / width,
                y1 / height,
                (x1 - x0) / width,
                (y1 - y0) / height,
                r["index"] / max(r["count"], 1),
                min(len(text), 200) / 200,
                sum(c.isdigit() for c in text) / max(len(text), 1),
                sum(c.isupper() for c in text) / max(len(text), 1),
                bool(amounts(text)),
                bool(receipt_date(text)),
                bool(re.search(r"\btotal\b", text, re.I)),
                bool(re.search(r"\b(subtotal|cash|change|tax|gst|sst)\b", text, re.I)),
            ]
        )
    return np.asarray(rows, dtype=float)


def make_model(c=1.0):
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text",
                            Pipeline(
                                [
                                    ("select", FunctionTransformer(line_texts)),
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            analyzer="char_wb",
                                            ngram_range=(3, 5),
                                            min_df=2,
                                            max_features=40000,
                                            sublinear_tf=True,
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        (
                            "geometry",
                            Pipeline(
                                [
                                    ("select", FunctionTransformer(line_geometry)),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(C=c, max_iter=1500, class_weight="balanced"),
            ),
        ]
    )
