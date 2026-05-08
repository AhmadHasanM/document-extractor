"""
app/core/detector.py
Deteksi tipe PDF: digital, scanned, atau mixed (per halaman)
"""
import fitz
from enum import Enum
from dataclasses import dataclass


class PDFType(Enum):
    DIGITAL = "digital"
    SCANNED = "scanned"
    MIXED = "mixed"


@dataclass
class PageTypeInfo:
    page_num: int          # 1-indexed
    pdf_type: PDFType
    text_length: int
    has_images: bool


class PDFDetector:
    """Deteksi tipe PDF dengan analisis per-halaman."""

    TEXT_THRESHOLD = 80   # karakter minimum agar halaman dianggap digital
    DIGITAL_RATIO = 0.75  # ≥75% halaman digital → DIGITAL
    SCANNED_RATIO = 0.25  # ≤25% halaman digital → SCANNED

    def detect(self, pdf_path: str) -> PDFType:
        page_types = self.get_page_types(pdf_path)
        total = len(page_types)
        if total == 0:
            return PDFType.DIGITAL
        digital_count = sum(1 for p in page_types if p.pdf_type == PDFType.DIGITAL)
        ratio = digital_count / total
        if ratio >= self.DIGITAL_RATIO:
            return PDFType.DIGITAL
        elif ratio <= self.SCANNED_RATIO:
            return PDFType.SCANNED
        return PDFType.MIXED

    def get_page_types(self, pdf_path: str) -> list[PageTypeInfo]:
        doc = fitz.open(pdf_path)
        results = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            images = page.get_images(full=False)
            is_digital = len(text) >= self.TEXT_THRESHOLD
            results.append(PageTypeInfo(
                page_num=i + 1,
                pdf_type=PDFType.DIGITAL if is_digital else PDFType.SCANNED,
                text_length=len(text),
                has_images=len(images) > 0,
            ))
        doc.close()
        return results