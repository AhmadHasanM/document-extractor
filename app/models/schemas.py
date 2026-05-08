"""
app/models/schemas.py
Pydantic models untuk request/response API
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class PDFTypeEnum(str, Enum):
    digital = "digital"
    scanned = "scanned"
    mixed   = "mixed"


class StatusEnum(str, Enum):
    pending    = "pending"
    processing = "processing"
    done       = "done"
    failed     = "failed"


# ──────────────────────────────────────────────
# Response: POST /extract
# ──────────────────────────────────────────────

class ExtractResponse(BaseModel):
    job_id: str = Field(..., description="ID unik job ekstraksi")
    pdf_type: PDFTypeEnum = Field(..., description="Tipe PDF yang terdeteksi")
    total_pages: int = Field(..., description="Jumlah halaman dokumen")
    total_chunks: int = Field(..., description="Jumlah chunk hasil segmentasi")
    download_url: str = Field(..., description="URL untuk mengunduh hasil Markdown")
    message: str = Field(default="Ekstraksi berhasil")


# ──────────────────────────────────────────────
# Response: GET /status/{job_id}  (opsional, untuk future async job)
# ──────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    job_id: str
    status: StatusEnum
    progress: Optional[int] = Field(None, description="Persentase progress 0–100")
    total_pages: Optional[int] = None
    processed_pages: Optional[int] = None
    error: Optional[str] = Field(None, description="Pesan error jika status=failed")


# ──────────────────────────────────────────────
# Response: GET /health
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


# ──────────────────────────────────────────────
# Internal: representasi chunk untuk serialisasi
# ──────────────────────────────────────────────

class ChunkSchema(BaseModel):
    chunk_id: str
    title: str
    level: int = Field(..., description="Level heading: 1=bab, 2=sub-bab, dst")
    content: str
    page_start: int
    page_end: int
    has_tables: bool = False
    has_formulas: bool = False
    has_images: bool = False

    @classmethod
    def from_chunk(cls, chunk) -> "ChunkSchema":
        return cls(
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            level=chunk.level,
            content=chunk.content,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            has_tables=bool(chunk.tables),
            has_formulas=bool(chunk.formulas),
            has_images=bool(chunk.images),
        )


# ──────────────────────────────────────────────
# Response: GET /preview/{job_id}  (opsional)
# Menampilkan ringkasan chunks tanpa konten penuh
# ──────────────────────────────────────────────

class PreviewResponse(BaseModel):
    job_id: str
    doc_title: str
    pdf_type: PDFTypeEnum
    total_pages: int
    chunks: list[ChunkSchema]