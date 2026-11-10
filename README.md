# Document Intelligence

Local receipt extraction from PDF and images. Tesseract/embedded PDF text supplies lines and bounding boxes; a trained text-and-layout classifier identifies company, address, date and total. Every predicted field carries the observed quote and its page coordinates. A local API, a plain web demo and CSV export use the same extraction path.

This is a measured portfolio prototype for document AI, not an automated bookkeeping system. Results remain marked `review_required`.

## Install and run

Validated on Python 3.12/Linux. Install Tesseract with its English language data separately:

```bash
# Debian/Ubuntu
sudo apt-get install tesseract-ocr tesseract-ocr-eng
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.lock
pip install --no-deps -e .
```

Model binaries are intentionally excluded from Git. From a fresh checkout, download the pinned dataset and train once before running inference. This requires approximately 510 MB of dataset downloads; the separately distributed original archive includes a trained artifact for local use. Only load trusted training artifacts.

```bash
document-intelligence download
document-intelligence train
document-intelligence extract examples/fictional_receipt.png
document-intelligence extract examples/fictional_receipt.png --format csv > fields.csv
uvicorn document_intelligence.api:app --host 127.0.0.1 --port 8003
```

Open `http://127.0.0.1:8003` for the upload demo or `/docs` for the API. The upload API accepts **raw file bytes**, not multipart form data:

```bash
curl -X POST http://127.0.0.1:8003/extract \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @examples/fictional_receipt.png
# Add ?format=csv for a downloadable CSV response.
```

The fictional sample has an accompanying expected-fields JSON file. It is a functional fixture, not part of the SROIE benchmark. The API does not save uploads or extracted text to disk.

## Reproduce training and evaluation

```bash
document-intelligence evaluate --ocr-count 40
```

The downloader verifies pinned hashes for approximately 510 MB of SROIE parquet files. The dataset has 626 official training and 361 test receipts in this pinned extended snapshot. Development partitions group training receipts by normalized vendor name: 523 fit and 103 validation receipts, with no vendor-group overlap. Two regularization settings are compared on validation. The selected classifier is frozen before the reported evaluation run; official test images are not training examples.

Physical-row assembly aligns the fragmented provided annotations with the full rows emitted by OCR. Training region labels are **weak labels derived from entity strings**, not official token-level annotations. The model combines character TF-IDF with normalized bounding boxes, row position, digit/case proportions, amount/date patterns and total/payment keywords. Logistic regression learns five classes, including `other`.

## Evaluation scope

The checked-in [evaluation report](reports/evaluation.json) separates two tracks:

| Track | Inputs | Size | Purpose |
| --- | --- | ---: | --- |
| Annotated-text KIE | Provided transcription and boxes, assembled into rows | 361 receipts | Measure extraction without running OCR |
| Tesseract end to end | Actual receipt images through the same ingestion/extraction API path | 40 fixed receipts | Include recognition and extraction errors |

The OCR subset is selected by the lowest SHA256 values of receipt IDs, independent of labels. Failed inputs count as wrong; they are not removed from the denominator. See [the measured results](reports/RESULTS.md) for actual field scores and [the model card](docs/MODEL_CARD.md) for limitations. Normalized exact match is project-defined and is not the official SROIE leaderboard metric. The two tracks have different sample sizes and are not a controlled paired estimate of OCR degradation.

The initial region-based experiment is retained in `reports/initial_region_evaluation.json`. Public reports contain aggregate metrics and counts only; running evaluation locally produces receipt-level diagnostics. Row assembly and bounded image downsampling were engineering corrections after inspecting that initial run; final test numbers are descriptive project results, not an untouched benchmark claim. Hyperparameter selection uses only the grouped development partition.

## Evidence, validation and input bounds

- Company/address values are selected from observed rows. Dates become ISO dates; totals become decimal strings. Fields without a sufficiently scored candidate are returned as null.
- Field scores are uncalibrated line-class scores. Evidence includes quote, box, page, source and OCR confidence where available.
- Subtotal/tax are separate keyword rules, tested on fictional fixtures only. Their arithmetic check uses `Decimal`; no tax extraction benchmark is claimed.
- The service accepts PDF, PNG, JPEG, single-frame TIFF and WEBP up to 8 MiB; PDFs have at most five unencrypted pages. Image inputs are capped at 40 megapixels and downsampled to at most three megapixels for OCR. OCR boxes are mapped back to the oriented input coordinates. PDF embedded-text boxes use page points; OCR on a rendered PDF uses rendered-image pixels, as identified by the evidence source.
- Per-page OCR has a 20-second timeout. CSV strings beginning with spreadsheet formula prefixes are escaped.

The benchmark covers Malaysian English receipts. Greek OCR, Greek invoices, line-item tables, handwriting, tax compliance and arbitrary business documents are not evaluated. Embedded PDF text is supported and tested on fixtures; the public benchmark uses images. Do not describe this as a multilingual invoice model.

## Verify and package

```bash
ruff check src tests
pytest
python -m build
docker build -t document-intelligence .
```

Tests include real Tesseract OCR on an authored image when Tesseract is installed, embedded-text PDF ingestion, document/page limits, date and money normalization, quote/box evidence, arithmetic checks, layout assembly, weak-label alignment and CSV injection protection. Unit tests do not download SROIE. Docker is supplied as configuration; it was not executed in the recorded host validation. Authentication, multi-user workload isolation and production load testing remain outside this local prototype.
