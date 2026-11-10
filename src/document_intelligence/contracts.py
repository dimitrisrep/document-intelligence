from dataclasses import asdict, dataclass

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 40_000_000
OCR_PIXELS = 3_000_000
MAX_PAGES = 5
FIELDS = ("company", "date", "address", "total")


@dataclass(frozen=True)
class Line:
    text: str
    bbox: tuple[float, float, float, float]
    width: float
    height: float
    page: int = 1
    confidence: float | None = None
    source: str = "ocr"

    def to_record(self, index=0, count=1):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Page dimensions must be positive.")
        x0, y0, x1, y1 = self.bbox
        if not 0 <= x0 <= x1 <= self.width or not 0 <= y0 <= y1 <= self.height:
            raise ValueError("Line box lies outside its page.")
        return {**asdict(self), "index": index, "count": count}
