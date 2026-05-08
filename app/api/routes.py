"""
app/api/routes.py
Semua endpoint API dipisah dari main.py agar lebih rapi
"""

import os
import shutil
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse

from app.core.detector import PDFDetector, PDFType
from app.core.chunker import SemanticChunker
from app.core.markdown import MarkdownGenerator
from app.core.processor import DocumentProcessor
from app.models.schemas import (
    ExtractResponse,
    HealthResponse,
    JobStatusResponse,
    PreviewResponse,
    ChunkSchema,
    PDFTypeEnum,
    StatusEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job store (ganti Redis jika butuh persistent)
# Format: { job_id: { "status": str, "meta": dict } }
_jobs: dict[str, dict] = {}

detector  = PDFDetector()
chunker   = SemanticChunker(use_semantic=True)
md_gen    = MarkdownGenerator()


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _cleanup(path: str):
    try:
        os.remove(path)
        logger.info(f"[cleanup] File dihapus: {path}")
    except FileNotFoundError:
        pass


def _require_job(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' tidak ditemukan")
    return _jobs[job_id]


# ──────────────────────────────────────────────
# POST /extract
# ──────────────────────────────────────────────

@router.post(
    "/extract",
    response_model=ExtractResponse,
    summary="Upload PDF dan ekstrak ke Markdown",
    description=(
        "Upload file PDF (digital maupun scan). "
        "Sistem akan otomatis mendeteksi tipe PDF, "
        "mengekstrak konten, dan mengembalikan URL download Markdown."
    ),
)
async def extract_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="File PDF yang akan diekstrak"),
    language: str = Query(
        default="ind+eng",
        description="Bahasa OCR (Tesseract lang code). Contoh: 'ind+eng', 'eng'",
    ),
    use_ai: bool = Query(
        default=True,
        description="Aktifkan Gemini AI untuk memperkaya hasil ekstraksi",
    ),
):
    # Validasi ekstensi
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang didukung")

    # Validasi ukuran (maks 500 MB)
    MAX_SIZE = 500 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Ukuran file melebihi batas 500 MB")

    # Simpan ke disk
    job_id  = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"

    with open(pdf_path, "wb") as f:
        f.write(content)

    # Daftarkan job
    _jobs[job_id] = {"status": StatusEnum.processing, "meta": {}}

    try:
        # 1. Deteksi tipe PDF
        pdf_type = detector.detect(str(pdf_path))
        logger.info(f"[{job_id}] Tipe PDF: {pdf_type.value}")

        # 2. Proses dokumen (ekstrak + chunk + AI enrich)
        processor = DocumentProcessor(str(pdf_path), language)

        if use_ai:
            chunks = processor.process()          # ekstrak + chunk + AI enrich
        else:
            # Tanpa AI: ekstrak & chunk saja
            if pdf_type == PDFType.DIGITAL:
                pages = processor._extract_digital()
            elif pdf_type == PDFType.SCANNED:
                pages = processor._extract_scanned()
            else:
                pages = processor._extract_mixed()
            chunks = chunker.chunk(pages)

        # 3. Generate Markdown
        doc_title = Path(file.filename).stem
        markdown_content = md_gen.generate(chunks, doc_title)

        # 4. Simpan output
        output_path = OUTPUT_DIR / f"{job_id}.md"
        output_path.write_text(markdown_content, encoding="utf-8")

        # Hitung total halaman dari chunks
        total_pages = max((c.page_end for c in chunks), default=0)

        # Update job store
        _jobs[job_id] = {
            "status": StatusEnum.done,
            "meta": {
                "pdf_type": pdf_type.value,
                "total_pages": total_pages,
                "total_chunks": len(chunks),
                "doc_title": doc_title,
                "chunks": [ChunkSchema.from_chunk(c).model_dump() for c in chunks],
            },
        }

        # Hapus file upload di background
        background_tasks.add_task(_cleanup, str(pdf_path))

        return ExtractResponse(
            job_id=job_id,
            pdf_type=PDFTypeEnum(pdf_type.value),
            total_pages=total_pages,
            total_chunks=len(chunks),
            download_url=f"/download/{job_id}",
            message="Ekstraksi berhasil",
        )

    except Exception as e:
        _jobs[job_id]["status"] = StatusEnum.failed
        _jobs[job_id]["meta"]["error"] = str(e)
        background_tasks.add_task(_cleanup, str(pdf_path))
        logger.error(f"[{job_id}] Gagal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Gagal memproses dokumen: {str(e)}")


# ──────────────────────────────────────────────
# GET /download/{job_id}
# ──────────────────────────────────────────────

@router.get(
    "/download/{job_id}",
    summary="Download hasil Markdown",
)
async def download_result(job_id: str):
    _require_job(job_id)

    output_path = OUTPUT_DIR / f"{job_id}.md"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="File output belum tersedia atau sudah dihapus")

    job_meta = _jobs[job_id].get("meta", {})
    filename  = f"{job_meta.get('doc_title', job_id)}.md"

    return FileResponse(
        path=str(output_path),
        media_type="text/markdown",
        filename=filename,
    )


# ──────────────────────────────────────────────
# GET /status/{job_id}
# ──────────────────────────────────────────────

@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Cek status job ekstraksi",
)
async def get_status(job_id: str):
    job = _require_job(job_id)
    meta = job.get("meta", {})

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        total_pages=meta.get("total_pages"),
        processed_pages=meta.get("total_pages") if job["status"] == StatusEnum.done else None,
        error=meta.get("error"),
    )


# ──────────────────────────────────────────────
# GET /preview/{job_id}
# ──────────────────────────────────────────────

@router.get(
    "/preview/{job_id}",
    response_model=PreviewResponse,
    summary="Lihat ringkasan chunks tanpa download",
)
async def preview_result(job_id: str):
    job = _require_job(job_id)

    if job["status"] != StatusEnum.done:
        raise HTTPException(
            status_code=400,
            detail=f"Job belum selesai. Status saat ini: {job['status']}",
        )

    meta   = job["meta"]
    chunks = [ChunkSchema(**c) for c in meta.get("chunks", [])]

    return PreviewResponse(
        job_id=job_id,
        doc_title=meta.get("doc_title", "Dokumen"),
        pdf_type=PDFTypeEnum(meta["pdf_type"]),
        total_pages=meta["total_pages"],
        chunks=chunks,
    )


# ──────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
def health_check():
    return HealthResponse(status="ok", version="1.0.0")


# ──────────────────────────────────────────────
# DELETE /job/{job_id}
# ──────────────────────────────────────────────

@router.delete(
    "/job/{job_id}",
    summary="Hapus job dan file output",
)
async def delete_job(job_id: str):
    _require_job(job_id)

    output_path = OUTPUT_DIR / f"{job_id}.md"
    if output_path.exists():
        output_path.unlink()

    del _jobs[job_id]

    return {"message": f"Job '{job_id}' berhasil dihapus"}