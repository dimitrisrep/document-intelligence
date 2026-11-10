"""Bounded local image/PDF ingestion. No file-provided links or scripts are executed."""

import io
from collections import defaultdict

import fitz
import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError

from document_intelligence.contracts import (
    MAX_BYTES,
    MAX_PAGES,
    MAX_PIXELS,
    OCR_PIXELS,
    Line,
)


def ocr_image(image, page=1):
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image exceeds the 40-megapixel input limit.")
    if image.width < 20 or image.height < 20:
        raise ValueError("Image is too small to read.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Coordinates in the result refer to the oriented input, before OCR downsampling.
    original_width, original_height = image.size
    if image.width * image.height > OCR_PIXELS:
        scale = (OCR_PIXELS / (image.width * image.height)) ** 0.5
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    sx, sy = original_width / image.width, original_height / image.height
    data = pytesseract.image_to_data(
        ImageOps.autocontrast(ImageOps.grayscale(image)),
        lang="eng",
        config="--psm 6",
        output_type=pytesseract.Output.DICT,
        timeout=20,
    )
    grouped = defaultdict(list)
    for i, value in enumerate(data["text"]):
        value = value.strip()
        if value and float(data["conf"][i]) >= 20:
            grouped[
                (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            ].append(i)
    result = []
    for indices in grouped.values():
        left = min(data["left"][i] for i in indices)
        top = min(data["top"][i] for i in indices)
        right = max(data["left"][i] + data["width"][i] for i in indices)
        bottom = max(data["top"][i] + data["height"][i] for i in indices)
        result.append(
            Line(
                " ".join(data["text"][i] for i in indices),
                (left * sx, top * sy, right * sx, bottom * sy),
                original_width,
                original_height,
                page,
                sum(float(data["conf"][i]) for i in indices) / len(indices),
                "tesseract_eng",
            )
        )
    return sorted(result, key=lambda line: (line.page, line.bbox[1], line.bbox[0]))


def pdf_lines(content):
    result = []
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("Invalid PDF.") from exc
    with document:
        if document.needs_pass or not 1 <= len(document) <= MAX_PAGES:
            raise ValueError("PDF must be unencrypted and contain 1-5 pages.")
        for number, page in enumerate(document, 1):
            rect = page.rect
            if (
                rect.width <= 0
                or rect.height <= 0
                or rect.width * rect.height * 2.5**2 > MAX_PIXELS
            ):
                raise ValueError("PDF page exceeds the rendering limit.")
            words = page.get_text("words")
            if sum(len(w[4]) for w in words) >= 20:
                grouped = defaultdict(list)
                for word in words:
                    grouped[(word[5], word[6])].append(word)
                for row in grouped.values():
                    row = sorted(row, key=lambda w: w[0])
                    result.append(
                        Line(
                            " ".join(w[4] for w in row),
                            (
                                min(w[0] for w in row),
                                min(w[1] for w in row),
                                max(w[2] for w in row),
                                max(w[3] for w in row),
                            ),
                            rect.width,
                            rect.height,
                            number,
                            None,
                            "pdf_embedded_text",
                        )
                    )
            else:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
                image = Image.frombytes(
                    "RGB", [pixmap.width, pixmap.height], pixmap.samples
                )
                result.extend(ocr_image(image, number))
    return result


def read_document(content: bytes):
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Document must contain 1 byte through 8 MiB.")
    if content.startswith(b"%PDF"):
        return pdf_lines(content)
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in {"PNG", "JPEG", "TIFF", "WEBP"}:
                raise ValueError("Supported inputs are PDF, PNG, JPEG, TIFF and WEBP.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image exceeds the 40-megapixel input limit.")
            if getattr(image, "n_frames", 1) > 1:
                raise ValueError("Multi-frame images must be supplied as a PDF.")
            return ocr_image(image)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("Invalid or unsupported image.") from exc
