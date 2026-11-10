import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation


def compact(text):
    return "".join(
        c for c in unicodedata.normalize("NFKC", str(text)).casefold() if c.isalnum()
    )


def money(text):
    """Return a two-decimal string, handling grouped US/EU decimal notation."""
    text = str(text).strip()
    if not re.fullmatch(r"(?:RM|MYR|EUR|USD|€|\$)?\s*[-+]?\d[\d., ]*", text, re.I):
        return None
    sign = "-" if "-" in text else ""
    digits = re.sub(r"[^\d.,]", "", text)
    if not digits:
        return None
    last = max(digits.rfind("."), digits.rfind(","))
    if last >= 0:
        if len(digits) - last - 1 != 2:
            return None
        digits = re.sub(r"[.,]", "", digits[:last]) + "." + digits[last + 1 :]
    try:
        value = Decimal(sign + digits)
        return (
            format(value.quantize(Decimal("0.01")), "f") if value.is_finite() else None
        )
    except InvalidOperation:
        return None


def amounts(text):
    return [
        m.group()
        for m in re.finditer(
            r"(?<!\d)(?:\d{1,3}(?:[., ]\d{3})+|\d+)[.,]\d{2}(?!\d)", text
        )
    ]


def receipt_date(text):
    match = re.search(
        r"\b(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b", text
    )
    if not match:
        return None
    value = re.sub(r"[.-]", "/", match.group())
    for fmt in ["%Y/%m/%d", "%d/%m/%Y", "%d/%m/%y"]:
        try:
            parsed = datetime.strptime(value, fmt).date()
            return parsed.isoformat() if 1990 <= parsed.year <= 2100 else None
        except ValueError:
            pass
    return None


def normalized_field(field, text):
    if text is None:
        return None
    if field == "total":
        return money(text)
    if field == "date":
        return receipt_date(text)
    return compact(text)
