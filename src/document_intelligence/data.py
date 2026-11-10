import hashlib
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq

from document_intelligence.contracts import Line
from document_intelligence.layout import assemble_rows

DATASET_ID = "jsdnrs/ICDAR2019-SROIE"
REVISION = "bffe40c26759f3376ec2b3ae9031dbba54cd587c"
CHECKSUMS = {
    "train": "b18c16b4d8481e5e4537a1700e4616907fe4acd92d6362a7e430b0e866213887",
    "test": "04f8f31b45944cc6e6459a7a95c851a721fc93ffec0a5c29ece9ded734a684c2",
}


def download(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for split, expected in CHECKSUMS.items():
        path = directory / f"{split}.parquet"
        if not path.exists():
            url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{REVISION}/data/{split}-00000-of-00001.parquet"
            temporary = path.with_suffix(".partial")
            with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            temporary.replace(path)
        verify(path, expected)


def verify(path: Path, expected: str):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("SROIE source checksum mismatch.")


def load_records(directory: Path, split: str, images=False):
    if split not in CHECKSUMS:
        raise ValueError("Unknown split.")
    path = directory / f"{split}.parquet"
    verify(path, CHECKSUMS[split])
    columns = ["key", "image_size", "entities", "words", "bboxes"] + (
        ["image"] if images else []
    )
    return pq.read_table(path, columns=columns).to_pylist()


def annotation_lines(record):
    return assemble_rows(
        [
            Line(
                text,
                tuple(box),
                record["image_size"]["width"],
                record["image_size"]["height"],
                source="provided_transcription",
            )
            for text, box in zip(record["words"], record["bboxes"])
        ]
    )
