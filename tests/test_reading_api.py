import io
import shutil

import fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from document_intelligence.api import create_app
from document_intelligence.contracts import MAX_BYTES
from document_intelligence.reading import ocr_image, read_document
from document_intelligence.service import DocumentExtractor


def test_pdf_embedded_text_has_page_coordinates():
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 60), "ACME SUPPLIES SDN BHD")
        page.insert_text((40, 100), "Date: 01/06/2024 Total: RM 12.00")
        content = doc.tobytes()
    lines = read_document(content)
    assert len(lines) == 2
    assert all(line.source == "pdf_embedded_text" and line.page == 1 for line in lines)
    for line in lines:
        line.to_record()  # The returned evidence boxes must be valid in page coordinates.


def test_pdf_page_limit_and_bad_files_are_enforced():
    with fitz.open() as doc:
        for _ in range(6):
            doc.new_page()
        content = doc.tobytes()
    with pytest.raises(ValueError, match="1-5"):
        read_document(content)
    for invalid in [b"", b"not an image", b"%PDF invalid", b"x" * (MAX_BYTES + 1)]:
        with pytest.raises(ValueError):
            read_document(invalid)


@pytest.mark.skipif(
    not shutil.which("tesseract"), reason="Tesseract is a separate system dependency"
)
def test_actual_tesseract_reads_generated_image():
    image = Image.new("RGB", (900, 220), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=42)
    draw.text((35, 25), "TOTAL RM 12.00", fill="black", font=font)
    draw.text((35, 105), "Date 01/06/2024", fill="black", font=font)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    lines = read_document(stream.getvalue())
    assert "12.00" in " ".join(line.text for line in lines)
    assert all(line.source == "tesseract_eng" for line in lines)


def test_downsampling_preserves_original_coordinate_system(monkeypatch):
    import document_intelligence.reading as module

    def bounded_ocr(image, **kwargs):
        assert image.width * image.height <= 3_000_000
        return {
            "text": ["TOTAL"],
            "conf": [95],
            "block_num": [1],
            "par_num": [1],
            "line_num": [1],
            "left": [image.width // 4],
            "top": [image.height // 4],
            "width": [image.width // 2],
            "height": [image.height // 2],
        }

    monkeypatch.setattr(module.pytesseract, "image_to_data", bounded_ocr)
    line = ocr_image(Image.new("RGB", (2400, 1600), "white"))[0]
    assert (line.width, line.height) == (2400, 1600)
    assert line.bbox[0] == pytest.approx(600, abs=2)
    assert line.bbox[2] == pytest.approx(1800, abs=3)
    line.to_record()


def test_api_missing_model_and_bounded_stream(tmp_path):
    with TestClient(create_app(DocumentExtractor(tmp_path))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/extract", content=b"test").status_code == 503
        assert (
            client.post("/extract", content=b"x" * (MAX_BYTES + 1)).status_code == 413
        )
        assert client.post("/extract?format=xml", content=b"test").status_code == 422


def test_raw_upload_routes_to_shared_service():
    class FakeService:
        model = object()

        def extract(self, content):
            assert content == b"receipt-content"
            return {"status": "review_required", "fields": {}}

    with TestClient(create_app(FakeService())) as client:
        result = client.post("/extract", content=b"receipt-content")
        assert (
            result.status_code == 200 and result.json()["status"] == "review_required"
        )
        csv = client.post("/extract?format=csv", content=b"receipt-content")
        assert "attachment" in csv.headers["content-disposition"]
