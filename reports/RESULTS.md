# Recorded field extraction results

Pinned SROIE snapshot: 361 test receipts; fixed, label-independent 40-image OCR subset. Results use project-normalized exact match.

| Field | Annotated text (361) | Tesseract end to end (40) |
| --- | ---: | ---: |
| company | 0.637 | 0.425 |
| date | 0.856 | 0.650 |
| address | 0.042 | 0.175 |
| total | 0.920 | 0.725 |
| Mean field exact match | 0.614 | 0.494 |
| All four fields correct | 0.014 | 0.000 |
| Micro F1 | 0.632 | 0.528 |

OCR ingestion failures: 0/40, counted as wrong. Public JSON reports contain aggregate metrics and failure counts; receipt-level keys and field values are excluded.

Address exact match is weak and is a documented improvement area. Do not substitute company/date/total scores for complete-receipt success. Annotated-text and OCR tracks have different sample sizes; they are not paired.

The initial region-based test report is also retained. Row grouping, label-alignment fixes and input downsampling followed that observed run, so these final numbers describe an engineering iteration rather than an untouched external benchmark. Regularization is chosen on grouped validation only. No tax, Greek-language, handwritten-text or line-item benchmark was run.
