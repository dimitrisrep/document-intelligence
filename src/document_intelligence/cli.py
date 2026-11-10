import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Receipt OCR and learned field extraction"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["download", "train", "evaluate"]:
        sub = commands.add_parser(name)
        sub.add_argument("--data", type=Path, default=Path("data/raw"))
        if name != "download":
            sub.add_argument("--artifacts", type=Path, default=Path("artifacts"))
        if name == "evaluate":
            sub.add_argument("--ocr-count", type=int, default=40)
    extract = commands.add_parser("extract")
    extract.add_argument("file", type=Path)
    extract.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    extract.add_argument("--format", choices=["json", "csv"], default="json")
    args = parser.parse_args()
    if args.command == "download":
        from document_intelligence.data import download

        download(args.data)
        return
    if args.command == "train":
        from document_intelligence.training import train

        result = train(args.data, args.artifacts)
    elif args.command == "evaluate":
        from document_intelligence.training import evaluate

        if args.ocr_count < 0:
            parser.error("--ocr-count must be non-negative")
        result = evaluate(args.data, args.artifacts, args.ocr_count)
    else:
        from document_intelligence.contracts import MAX_BYTES
        from document_intelligence.service import DocumentExtractor, csv_export

        if args.file.stat().st_size > MAX_BYTES:
            parser.error("File exceeds 8 MiB.")
        result = DocumentExtractor(args.artifacts).extract(args.file.read_bytes())
        if args.format == "csv":
            print(csv_export(result), end="")
            return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
