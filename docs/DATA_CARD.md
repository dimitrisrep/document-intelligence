# Data card

Source: [jsdnrs/ICDAR2019-SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE), an extended/repaired distribution of the ICDAR 2019 Scanned Receipts OCR and Information Extraction dataset. The pinned revision is `bffe40c26759f3376ec2b3ae9031dbba54cd587c`. This snapshot contains 626 train and **361 test** receipts, including additional test annotations. It is not the commonly cited original 347-test configuration.

The dataset card declares **CC BY 4.0** and supplies LICENSE/NOTICE files. Attribution: ICDAR 2019 SROIE competition/dataset contributors and jsdnrs for this distribution. See [the original task repository](https://github.com/zzzDavid/ICDAR-2019-SROIE) and [the pinned dataset card](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE/blob/bffe40c26759f3376ec2b3ae9031dbba54cd587c/README.md). No endorsement by the dataset authors is implied.

Each receipt includes an image, text regions, pixel boxes and four string-valued entities: company, date, address and total. Receipt photographs include merchant names and transaction details. The downloadable source files are not redistributed with this code; source URLs and SHA256 verification are in `src/document_intelligence/data.py`.

Training assigns weak row labels by matching normalized/fuzzy text to entity strings. Physical rows join nearby text regions using geometry only. Labels are imperfect: repeated equal prices can match the total, annotations can omit parts of gold entity strings, and address boundaries are noisy. The model card and aggregate metrics in the experiment reports expose the consequences.

Training receipts are divided by normalized vendor name with seed 42. This prevents the same exact normalized training vendor from appearing in both fit and validation, but does not resolve spelling variants or remove vendors shared with the dataset's official test partition. Official test data is never appended to training.

The 40-image OCR sample is determined from receipt IDs before looking at labels. Public evaluation reports retain counts and metrics, but omit receipt-level keys and values. The authored sample under `examples/` is fictional and carries no real transaction or customer information.
