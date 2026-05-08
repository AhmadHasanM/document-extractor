import json
import logging
import os
import shutil
import uuid

from pathlib import Path

from dotenv import load_dotenv

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.responses import FileResponse

load_dotenv()

logging.basicConfig(level=logging.INFO)

from app.core.processor import DocumentProcessor
from app.core.markdown import MarkdownGenerator

BASE_OUTPUT = Path("outputs")

MARKDOWN_DIR = BASE_OUTPUT / "markdown"
IMAGE_DIR = BASE_OUTPUT / "images"
TABLE_DIR = BASE_OUTPUT / "tables"
JSON_DIR = BASE_OUTPUT / "json"

UPLOAD_DIR = Path("uploads")

for d in [
    MARKDOWN_DIR,
    IMAGE_DIR,
    TABLE_DIR,
    JSON_DIR,
    UPLOAD_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="AI Document Extractor",
    version="3.0.0",
)

markdown_gen = MarkdownGenerator()

logger = logging.getLogger(__name__)


@app.post("/extract")
async def extract_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = "ind+eng",
    use_ai: bool = False,
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Hanya PDF")

    job_id = str(uuid.uuid4())

    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"

    try:

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        processor = DocumentProcessor(
            str(pdf_path),
            language=language,
        )

        chunks = processor.process(
            use_ai=use_ai
        )

        markdown = markdown_gen.generate(
            chunks,
            doc_title=file.filename
        )


        md_path = MARKDOWN_DIR / f"{job_id}.md"

        md_path.write_text(
            markdown,
            encoding="utf-8"
        )

        json_output = []

        for chunk in chunks:

            json_output.append({

                "chunk_id": chunk.chunk_id,

                "title": chunk.title,

                "level": chunk.level,

                "content": chunk.content,

                "page_start": chunk.page_start,

                "page_end": chunk.page_end,

                "tables": len(chunk.tables),

                "images": chunk.images,
            })

        json_path = JSON_DIR / f"{job_id}.json"

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                json_output,
                f,
                ensure_ascii=False,
                indent=2
            )

        background_tasks.add_task(
            os.remove,
            pdf_path
        )

        return {
            "status": "success",
            "job_id": job_id,
            "markdown": f"/download/markdown/{job_id}",
            "json": f"/download/json/{job_id}",
        }

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            500,
            str(e)
        )

@app.get("/download/markdown/{job_id}")
async def download_markdown(job_id: str):

    path = MARKDOWN_DIR / f"{job_id}.md"

    if not path.exists():
        raise HTTPException(404)

    return FileResponse(
        path,
        media_type="text/markdown",
        filename=f"{job_id}.md"
    )


@app.get("/download/json/{job_id}")
async def download_json(job_id: str):

    path = JSON_DIR / f"{job_id}.json"

    if not path.exists():
        raise HTTPException(404)

    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{job_id}.json"
    )


@app.get("/")
def root():
    return {
        "message": "AI Document Extractor"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }