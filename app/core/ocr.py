"""
app/core/ocr.py

OCR pipeline untuk halaman scan:
- img2table → deteksi tabel
- Tesseract OCR
- False positive filtering
- Table masking
- Better table validation
"""

import io
import logging
import os
import re

from dataclasses import dataclass, field

import cv2
import fitz
import numpy as np
import pytesseract

from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class OCRPageContent:
    page_num: int
    text: str
    confidence: float = 0.0
    source: str = "tesseract"
    tables: list = field(default_factory=list)
    images: list = field(default_factory=list)


def preprocess_image(
    pil_img: Image.Image
) -> tuple:

    img = np.array(
        pil_img.convert("RGB")
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    gray = _deskew(gray)

    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=12,
        templateWindowSize=7,
        searchWindowSize=21
    )

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )

    return Image.fromarray(binary), denoised


def _deskew(gray: np.ndarray):

    try:

        edges = cv2.Canny(
            gray,
            50,
            150,
            apertureSize=3
        )

        lines = cv2.HoughLines(
            edges,
            1,
            np.pi / 180,
            threshold=100
        )

        if lines is None:
            return gray

        angles = []

        for line in lines[:20]:

            _, theta = line[0]

            angle = (
                theta - np.pi / 2
            ) * 180 / np.pi

            if abs(angle) < 10:
                angles.append(angle)

        if not angles:
            return gray

        median_angle = float(
            np.median(angles)
        )

        if abs(median_angle) < 0.5:
            return gray

        h, w = gray.shape

        M = cv2.getRotationMatrix2D(
            (w // 2, h // 2),
            median_angle,
            1.0
        )

        return cv2.warpAffine(
            gray,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    except Exception:
        return gray


def remove_table_regions(
    arr: np.ndarray,
    bboxes: list
) -> np.ndarray:

    result = arr.copy()

    for (x0, y0, x1, y1) in bboxes:

        px0 = max(0, int(x0) - 8)
        py0 = max(0, int(y0) - 8)

        px1 = min(result.shape[1], int(x1) + 8)
        py1 = min(result.shape[0], int(y1) + 8)

        result[py0:py1, px0:px1] = 255

    return result


def _is_valid_table(
    rows: list[list[str]]
) -> bool:

    """
    Validasi ketat agar image/layout/paragraf
    tidak dianggap tabel.
    """

    if not rows or len(rows) < 2:
        return False

    cleaned = []

    for r in rows:

        row = [
            str(c or "").strip()
            for c in r
        ]

        if any(cell for cell in row):
            cleaned.append(row)

    rows = cleaned

    if len(rows) < 2:
        return False

    max_cols = max(
        len(r)
        for r in rows
    )

    if max_cols < 2:
        return False

    total_cells = sum(
        len(r)
        for r in rows
    )

    if total_cells == 0:
        return False

    filled_cells = sum(
        1
        for r in rows
        for c in r
        if str(c).strip()
    )

    fill_ratio = filled_cells / total_cells

    if fill_ratio < 0.35:
        return False

    aligned_rows = 0

    for r in rows:

        non_empty = sum(
            1
            for c in r
            if str(c).strip()
        )

        if non_empty >= 2:
            aligned_rows += 1

    if aligned_rows < 2:
        return False

    long_cells = 0

    for r in rows:
        for c in r:

            words = str(c).split()

            if len(words) > 20:
                long_cells += 1

    if long_cells > len(rows):
        return False

    avg_words = (
        sum(
            len(str(c).split())
            for r in rows
            for c in r
        )
        / max(total_cells, 1)
    )

    if avg_words < 0.5:
        return False

    if len(rows) <= 2:

        giant_row = False

        for r in rows:

            joined = " ".join(
                str(c)
                for c in r
            )

            if len(joined.split()) > 30:
                giant_row = True

        if giant_row:
            return False

    single_col_rows = 0

    for r in rows:

        non_empty = sum(
            1
            for c in r
            if str(c).strip()
        )

        if non_empty <= 1:
            single_col_rows += 1

    if single_col_rows >= len(rows) * 0.7:
        return False

    return True


def _clean_table_rows(
    rows: list[list[str]]
) -> list[list[str]]:

    cleaned = []

    for row in rows:

        new_row = []

        for cell in row:

            val = (
                str(cell).strip()
                if str(cell) not in ("nan", "None", "NaN")
                else ""
            )

            val = re.sub(r"\n+", " ", val)

            val = re.sub(
                r"\s{2,}",
                " ",
                val
            ).strip()

            new_row.append(val)

        if any(c for c in new_row):
            cleaned.append(new_row)

    return cleaned


class ScannedTableExtractor:

    def __init__(self, lang: str = "ind+eng"):

        try:

            from img2table.ocr import TesseractOCR as Img2Tess
            from img2table.document import Image as Img2Image

            self._Img2Image = Img2Image

            self._ocr = Img2Tess(
                lang=lang
            )

            self._available = True

            logger.info(
                "[img2table] Ready"
            )

        except ImportError:

            logger.warning(
                "[img2table] unavailable"
            )

            self._available = False

    def extract(
        self,
        pil_img: Image.Image
    ) -> tuple:

        if not self._available:
            return [], []

        tables, bboxes = self._run(
            pil_img,
            borderless=False
        )

        if not tables:

            tables, bboxes = self._run(
                pil_img,
                borderless=True
            )

        return tables, bboxes

    def _run(
        self,
        pil_img: Image.Image,
        borderless: bool
    ) -> tuple:

        try:

            buf = io.BytesIO()

            pil_img.save(
                buf,
                format="PNG"
            )

            buf.seek(0)

            doc = self._Img2Image(
                src=buf
            )

            extracted = doc.extract_tables(
                ocr=self._ocr,
                implicit_rows=True,
                borderless_tables=borderless,
                min_confidence=60 if borderless else 45,
            )

        except Exception as e:

            logger.warning(
                f"[img2table borderless={borderless}] {e}"
            )

            return [], []

        tables = []
        bboxes = []

        for t in extracted:

            df = t.df

            if df is None or df.empty:
                continue

            header = [
                str(c)
                for c in df.columns.tolist()
            ]

            if all(h.isdigit() for h in header):

                rows = [
                    list(r)
                    for r in df.values.tolist()
                ]

            else:

                rows = [header] + [
                    list(r)
                    for r in df.values.tolist()
                ]

            rows = _clean_table_rows(rows)

            if not _is_valid_table(rows):
                continue

            try:

                bbox = (
                    t.bbox.x1,
                    t.bbox.y1,
                    t.bbox.x2,
                    t.bbox.y2,
                )

                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]

                # reject giant layout block
                if (
                    bw > pil_img.width * 0.9
                    and bh > pil_img.height * 0.5
                ):
                    continue

            except Exception:

                bbox = (
                    0,
                    0,
                    pil_img.width,
                    pil_img.height,
                )

            try:

                from .table_repair import repair_table

                rows = repair_table(rows)

            except Exception:
                pass

            if not rows or len(rows) < 2:
                continue

            tables.append(rows)
            bboxes.append(bbox)

            logger.info(
                f"[img2table] VALID table: "
                f"{len(rows)} rows x "
                f"{max(len(r) for r in rows)} cols"
            )

        return tables, bboxes


class TesseractOCR:

    def __init__(self, lang: str = "ind+eng"):

        self.lang = lang

        self.table_extractor = (
            ScannedTableExtractor(lang=lang)
        )

        logger.info(
            f"[TesseractOCR] Ready lang={lang}"
        )

    def extract_all(
        self,
        pdf_path: str,
        dpi: int = 300
    ) -> list:

        from .extractor import TableData

        doc = fitz.open(pdf_path)

        results = []

        for i, page in enumerate(doc):

            logger.info(
                f"[OCR] Page {i+1}/{len(doc)}"
            )

            pix = page.get_pixmap(
                matrix=fitz.Matrix(
                    dpi / 72,
                    dpi / 72
                )
            )

            pil_img = Image.open(
                io.BytesIO(
                    pix.tobytes("png")
                )
            )

            result = self._process_page(
                pil_img,
                i + 1
            )

            results.append(result)

        doc.close()

        return results

    def _process_page(
        self,
        pil_img: Image.Image,
        page_num: int
    ):

        from .extractor import TableData

        binary_img, clean_gray = preprocess_image(
            pil_img
        )

        table_rows_list, table_bboxes = (
            self.table_extractor.extract(pil_img)
        )

        if table_bboxes:

            masked = remove_table_regions(
                np.array(binary_img),
                table_bboxes
            )

            text_img = Image.fromarray(masked)

        else:

            text_img = binary_img

        text, conf = self._run_tesseract(
            text_img
        )

        tables = [
            TableData(
                rows=r,
                page_num=page_num
            )
            for r in table_rows_list
        ]

        return OCRPageContent(
            page_num=page_num,
            text=text,
            confidence=conf,
            source="tesseract+img2table",
            tables=tables,
        )

    def _run_tesseract(
        self,
        img: Image.Image
    ) -> tuple:

        config = (
            f"--oem 3 --psm 6 -l {self.lang}"
        )

        try:

            text = pytesseract.image_to_string(
                img,
                config=config
            )

        except Exception as e:

            logger.warning(
                f"Tesseract error: {e}"
            )

            return "", 0.0

        try:

            data = pytesseract.image_to_data(
                img,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            confs = [
                int(c)
                for c in data["conf"]
                if str(c).lstrip("-").isdigit()
                and int(c) > 0
            ]

            conf = (
                sum(confs) / len(confs) / 100
                if confs else 0.7
            )

        except Exception:

            conf = 0.7

        return _clean_ocr_text(text), conf


def _clean_ocr_text(text: str) -> str:

    lines = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        ratio = (
            sum(
                c.isalnum()
                or c in " .,;:!?()-/"
                for c in line
            )
            / max(len(line), 1)
        )

        if ratio < 0.3 and len(line) > 3:
            continue

        if re.search(
            r"camscanner|dipindai dengan|scan by easy",
            line,
            re.IGNORECASE
        ):
            continue

        lines.append(line)

    merged = []

    for line in lines:

        if (
            merged
            and line
            and len(merged[-1]) < 65
            and line[0].islower()
            and merged[-1][-1] not in ".!?:"
        ):

            merged[-1] += " " + line

        else:

            merged.append(line)

    return "\n".join(merged).strip()


def get_ocr_extractor(
    lang: str = "ind+eng"
):

    return TesseractOCR(lang=lang)