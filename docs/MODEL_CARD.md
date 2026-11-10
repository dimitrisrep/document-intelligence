# Model card

## Method

The learned component is balanced logistic regression over character 3-5 gram TF-IDF and standardized spatial/text-pattern features. It predicts company/address/date/total/other for observed rows. The pipeline learns its TF-IDF vocabulary and scaling from the fit partition only. Two values of `C` (0.5 and 2.0) are compared on grouped validation field exact match. Public training metadata records the selected value, split counts and weak-label counts; receipt IDs are excluded.

The recognizer is local Tesseract English (`--psm 6`), not a neural document model fine-tuned by this project. Text-based PDFs use PyMuPDF's embedded text when available; scanned PDFs render to images. No external generative API is called and missing values are not generated. The classifier is trained on provided transcriptions, so OCR misspellings and different row segmentation introduce domain shift.

## Evaluation interpretation

Read `reports/RESULTS.md` and `reports/evaluation.json` together. Normalization removes case/punctuation for company/address, validates numeric dates and compares exact decimal totals. Each missing or incorrect field counts as an error. Mean field exact match weights the four fields equally. The public report also contains micro precision/recall/F1, all-fields-correct receipt rate and aggregate failure counts.

Strict address exact match is a major weakness. Partial address detection must not be presented as complete extraction. Small variations or omitted address text can fail the exact metric; a value must still be verified against the document. The 40-image end-to-end estimate is small and has no confidence interval. The initially observed test run informed engineering corrections to segmentation and input handling, so the final result is descriptive rather than a pristine model-selection benchmark.

Tax/subtotal parsing is a separate label rule. Only company/date/address/total are benchmarked. No Greek OCR benchmark, invoice line-item benchmark, compliance evaluation or commercial deployment was performed. Every response is marked `review_required`, independently of score.

## Operational contract

Joblib artifacts execute Python during loading: accept only trusted locally trained artifacts. The service verifies schema and scikit-learn version. Uploaded bytes are bounded before buffering; the source file is not written to disk. OCR timeouts and image/page limits reduce resource exposure but are not a sandbox for hostile parsers or a substitute for deployment isolation.

Evidence boxes refer to the coordinate system associated with their source. PNG/JPEG OCR coordinates use the EXIF-oriented image size, before downsampling. Embedded PDF text uses page points; scanned-PDF OCR uses the rendered page image size. The CLI/API return the same learned-field results. Currency is a limited observed-symbol rule (MYR/EUR), not an inferred country or exchange rate. Financial amounts are strings to preserve decimal precision.
