"""
app/core/chunker.py

Semantic chunker:
- Heading based chunking
- Formula support
- Image support
- Inject image near closest paragraph
"""

import re
import uuid

from dataclasses import dataclass, field
from typing import Optional


# =========================================================
# DOCUMENT CHUNK
# =========================================================
@dataclass
class DocumentChunk:

    chunk_id: str

    title: str

    level: int

    content: str

    page_start: int

    page_end: int

    tables: list = field(default_factory=list)

    images: list = field(default_factory=list)

    formulas: list = field(default_factory=list)


# =========================================================
# HEADING PATTERNS
# =========================================================
_H1 = re.compile(
    r"^(BAB\s+[IVXLC\d]+|ABSTRAK|ABSTRACT|PENDAHULUAN|KESIMPULAN|"
    r"SIMPULAN|DAFTAR\s+PUSTAKA|KEPUSTAKAAN|DAFTAR\s+ISI|"
    r"KATA\s+PENGANTAR|HASIL\s+DAN\s+PEMBAHASAN|"
    r"BAHAN\s+DAN\s+METODE|METODE\s+PENELITIAN|METODOLOGI|"
    r"TINJAUAN\s+PUSTAKA|LANDASAN\s+TEORI|KERANGKA\s+TEORI|"
    r"KESIMPULAN\s+DAN\s+SARAN|SIMPULAN\s+DAN\s+SARAN|"
    r"SARAN|REKOMENDASI|LATAR\s+BELAKANG|RUMUSAN\s+MASALAH|"
    r"TUJUAN\s+PENELITIAN|MANFAAT\s+PENELITIAN)(\s|$)",
    re.IGNORECASE,
)

_H2 = re.compile(
    r"^(\d+\.\d+\.?\s+\S|"
    r"[A-Z]\.\s+[A-Z\d]|"
    r"\d+[\)\.]\s+[A-Z][a-z])",
)

_H3 = re.compile(
    r"^\d+\.\d+\.\d+\.?\s+\S",
)

HEADING_PATTERNS = [
    (1, _H1),
    (3, _H3),
    (2, _H2),
]


# =========================================================
# ANTI FALSE POSITIVE
# =========================================================
_ANTI = [
    re.compile(r"\d{4}[,\.]"),
    re.compile(r"@|\.com|\.id|http|www\.", re.I),
    re.compile(r"(Rineka|Pustaka|Salemba|Erlangga|Alfabeta|Prenada)", re.I),
    re.compile(r"^(NIP|NIK|No\.|Nomor)\s*[\.\:]", re.I),
    re.compile(r"^\d{9,}"),
]


_NOISE = [
    re.compile(r"^Scan by Easy Scanner", re.I),
    re.compile(r"^Dipindai dengan CamScanner", re.I),
    re.compile(r"^Jurnal\s+\w+.*Vol\w*", re.I),
    re.compile(r"^\d{1,3}\s*$"),
]


# =========================================================
# HELPERS
# =========================================================
def _is_noise(line: str):

    for p in _NOISE:

        if p.search(line):
            return True

    return False


def _is_anti_heading(line: str):

    for p in _ANTI:

        if p.search(line):
            return True

    return False


# =========================================================
# HEADING DETECTOR
# =========================================================
def detect_heading_level(
    line: str,
    font_headings: list | None = None
) -> Optional[int]:

    cleaned = line.strip()

    if (
        not cleaned
        or len(cleaned) < 3
        or _is_anti_heading(cleaned)
    ):
        return None

    # ======================================
    # FONT BASED DETECTION
    # ======================================
    if font_headings:

        for fh in font_headings:

            if fh.get("text", "").strip() == cleaned:

                size = fh.get("size", 0)

                is_bold = fh.get("bold", False)

                if size >= 13 or (size >= 11 and is_bold):

                    return 1 if size >= 13 else 2

    # ======================================
    # REGEX DETECTION
    # ======================================
    for level, pat in HEADING_PATTERNS:

        if pat.search(cleaned):
            return level

    # ======================================
    # ALL CAPS
    # ======================================
    words = cleaned.split()

    if (
        cleaned.isupper()
        and 2 <= len(words) <= 10
        and len(cleaned) > 5
        and not any(c.isdigit() for c in cleaned)
        and not _is_anti_heading(cleaned)
    ):
        return 1

    return None


# =========================================================
# SEMANTIC CHUNKER
# =========================================================
class SemanticChunker:

    def chunk(self, pages) -> list[DocumentChunk]:

        chunks: list[DocumentChunk] = []

        current: Optional[DocumentChunk] = None

        for page in pages:

            raw_text = getattr(page, "text", "") or ""

            font_headings = getattr(page, "headings", []) or []

            lines = raw_text.split("\n")

            for line in lines:

                line = line.strip()

                if not line or _is_noise(line):
                    continue

                level = detect_heading_level(
                    line,
                    font_headings
                )

                # ==================================
                # NEW HEADING
                # ==================================
                if level is not None:

                    if current is not None:
                        chunks.append(current)

                    current = DocumentChunk(
                        chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                        title=line,
                        level=level,
                        content="",
                        page_start=page.page_num,
                        page_end=page.page_num,
                    )

                # ==================================
                # NORMAL CONTENT
                # ==================================
                else:

                    if current is None:

                        current = DocumentChunk(
                            chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                            title="Pembuka",
                            level=1,
                            content="",
                            page_start=page.page_num,
                            page_end=page.page_num,
                        )

                    current.content += line + "\n"

                    current.page_end = page.page_num

            # ==================================
            # ATTACH TABLES
            # ==================================
            if current is not None:

                for tbl in getattr(page, "tables", []) or []:
                    current.tables.append(tbl)

                for img in getattr(page, "images", []) or []:
                    current.images.append(img)

                for formula in getattr(page, "formulas", []) or []:
                    current.formulas.append(formula)

        # ==================================
        # FLUSH
        # ==================================
        if current is not None:
            chunks.append(current)

        return self._finalize(chunks)

    # =====================================================
    # FINALIZE
    # =====================================================
    def _finalize(
        self,
        chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:

        if not chunks:
            return []

        result: list[DocumentChunk] = []

        i = 0

        while i < len(chunks):

            chunk = chunks[i]

            has_content = bool(chunk.content.strip())

            has_tables = bool(chunk.tables)

            if not has_content and not has_tables:

                if result:

                    result[-1].content += f"\n**{chunk.title}**\n"

                    result[-1].page_end = chunk.page_end

                elif i + 1 < len(chunks):

                    chunks[i + 1].content = (
                        f"**{chunk.title}**\n"
                        + chunks[i + 1].content
                    )

                    chunks[i + 1].page_start = chunk.page_start

            else:

                result.append(chunk)

            i += 1

        return result