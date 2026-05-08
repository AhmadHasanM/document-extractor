"""
app/core/processor.py
Orchestrator utama: deteksi tipe PDF → ekstrak → chunk → (optional AI enrich)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .detector import PDFDetector, PDFType
from .extractor import DigitalExtractor, PageContent
from .chunker import SemanticChunker, DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class UnifiedPage:
    page_num: int
    text: str
    source: str                  # "digital" | "tesseract" | "deepseek_vl"
    tables: list = field(default_factory=list)
    images: list = field(default_factory=list)
    headings: list = field(default_factory=list)
    ocr_confidence: Optional[float] = None


def _to_unified(page, source: str) -> UnifiedPage:
    return UnifiedPage(
        page_num=page.page_num,
        text=getattr(page, "text", ""),
        source=source,
        tables=getattr(page, "tables", []),
        images=getattr(page, "images", []),
        headings=getattr(page, "headings", []),
        ocr_confidence=getattr(page, "confidence", None),
    )

class GeminiEnricher:
    """Koreksi OCR/typo menggunakan Gemini Flash."""

    MODEL = "gemini-2.5-flash"
    MAX_RETRIES = 3
    COOLDOWN = 65

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self._model = genai.GenerativeModel(self.MODEL)
        self._calls = 0

    def enrich(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        logger.info(f"[Gemini] Enriching {len(chunks)} chunks...")
        for chunk in chunks:
            try:
                improved = self._call(chunk.content)
                if improved:
                    chunk.content = improved
            except Exception as e:
                logger.warning(f"[Gemini] chunk '{chunk.title[:40]}': {e}")
        return chunks

    def _call(self, text: str) -> str:
        self._calls += 1
        if self._calls > 1 and self._calls % 10 == 0:
            time.sleep(self.COOLDOWN)

        prompt = f"""Anda adalah proofreader jurnal akademik Indonesia.
Tugas:
- Perbaiki typo, ejaan salah, dan kalimat terpotong akibat OCR
- Pertahankan struktur paragraf dan heading asli
- JANGAN ubah fakta, angka, atau nama ilmiah
- Output HANYA teks yang sudah diperbaiki

TEKS:
{text[:12000]}"""

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = self._model.generate_content(prompt)
                return resp.text or ""
            except Exception as e:
                err = str(e).lower()
                if any(k in err for k in ["429", "quota", "rate"]) and attempt < self.MAX_RETRIES:
                    wait = self.COOLDOWN * attempt
                    logger.warning(f"[Gemini] Rate limit → wait {wait}s")
                    time.sleep(wait)
                else:
                    raise
        return ""

class DocumentProcessor:

    def __init__(self, pdf_path: str, language: str = "ind+eng"):
        self.pdf_path = pdf_path
        self.language = language
        self.detector = PDFDetector()
        self.chunker = SemanticChunker()

    def process(self, use_ai: bool = False) -> list[DocumentChunk]:
        pdf_type = self.detector.detect(self.pdf_path)
        logger.info(f"[Processor] PDF type: {pdf_type.value}")

        pages = self._extract_pages(pdf_type)
        logger.info(f"[Processor] Extracted {len(pages)} pages")

        chunks = self.chunker.chunk(pages)
        logger.info(f"[Processor] {len(chunks)} chunks created")

        if use_ai and os.getenv("GEMINI_API_KEY"):
            try:
                enricher = GeminiEnricher()
                chunks = enricher.enrich(chunks)
                logger.info("[Processor] AI enrichment done")
            except Exception as e:
                logger.warning(f"[Processor] AI enrichment failed: {e}")

        return chunks

    def _extract_pages(self, pdf_type: PDFType) -> list[UnifiedPage]:
        if pdf_type == PDFType.DIGITAL:
            return self._extract_digital()
        elif pdf_type == PDFType.SCANNED:
            return self._extract_scanned()
        else:
            return self._extract_mixed()

    def _extract_digital(self) -> list[UnifiedPage]:
        extractor = DigitalExtractor(self.pdf_path)
        pages = extractor.extract_all()
        return [_to_unified(p, "digital") for p in pages]

    def _extract_scanned(self) -> list[UnifiedPage]:
        from .ocr import get_ocr_extractor
        ocr = get_ocr_extractor(self.language)
        pages = ocr.extract_all(self.pdf_path)
        return [_to_unified(p, p.source) for p in pages]

    def _extract_mixed(self) -> list[UnifiedPage]:
        """Per-halaman: digital pakai DigitalExtractor, scanned pakai OCR."""
        page_types = self.detector.get_page_types(self.pdf_path)
        digital_extractor = DigitalExtractor(self.pdf_path)
        digital_pages = {p.page_num: p for p in digital_extractor.extract_all()}

        # OCR hanya untuk halaman scanned
        scanned_nums = [p.page_num for p in page_types if p.pdf_type.value == "scanned"]
        ocr_pages: dict[int, UnifiedPage] = {}

        if scanned_nums:
            from .ocr import get_ocr_extractor
            import fitz
            ocr = get_ocr_extractor(self.language)
            # Extract semua lalu filter
            all_ocr = {p.page_num: p for p in ocr.extract_all(self.pdf_path)}
            for num in scanned_nums:
                if num in all_ocr:
                    ocr_pages[num] = _to_unified(all_ocr[num], all_ocr[num].source)

        unified = []
        for info in page_types:
            pnum = info.page_num
            if info.pdf_type.value == "digital" and pnum in digital_pages:
                unified.append(_to_unified(digital_pages[pnum], "digital"))
            elif pnum in ocr_pages:
                unified.append(ocr_pages[pnum])
            else:
                # Fallback: empty page
                unified.append(UnifiedPage(
                    page_num=pnum, text="", source="empty"
                ))

        return unified