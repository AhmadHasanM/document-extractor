"""
app/core/extractor.py

Digital PDF extractor dengan:
- Multi-column layout detection
- Rect-grid based table extraction
- Heading detection via font size
- False-positive table filtering
- Image extraction + bbox
"""

import re
import fitz
import pdfplumber
import statistics

from dataclasses import dataclass, field


# ============================================================
# TABLE DATA
# ============================================================

@dataclass
class TableData:
    rows: list[list[str]]
    caption: str = ""
    page_num: int = 0
    bbox: tuple = field(default_factory=tuple)

    def to_markdown(self) -> str:
        if not self.rows:
            return ""

        rows = []

        header = [
            str(c or "").strip().replace("\n", " ").replace("|", "\\|")
            for c in self.rows[0]
        ]

        rows.append("| " + " | ".join(header) + " |")
        rows.append("| " + " | ".join(["---"] * len(header)) + " |")

        for row in self.rows[1:]:

            cells = [
                str(c or "").strip().replace("\n", " ").replace("|", "\\|")
                for c in row
            ]

            while len(cells) < len(header):
                cells.append("")

            rows.append("| " + " | ".join(cells[:len(header)]) + " |")

        return "\n".join(rows)


# ============================================================
# PAGE CONTENT
# ============================================================

@dataclass
class PageContent:
    page_num: int
    text: str
    tables: list[TableData] = field(default_factory=list)
    headings: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)


# ============================================================
# DIGITAL EXTRACTOR
# ============================================================

class DigitalExtractor:

    HEADER_FONT_SIZE = 12.0
    BODY_FONT_SIZE = 9.0

    NOISE_PATTERNS = [
        r"^Scan by Easy Scanner",
        r"^Jurnal\s+\w+.*Vol\w*",
        r"^\d+\s*$",
    ]

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    # ============================================================
    # MAIN
    # ============================================================

    def extract_all(self) -> list[PageContent]:

        results = []

        doc = fitz.open(self.pdf_path)

        with pdfplumber.open(self.pdf_path) as plumber_pdf:

            for i, page in enumerate(doc):

                plumber_page = plumber_pdf.pages[i]

                content = self._extract_page(
                    page,
                    plumber_page,
                    i + 1
                )

                results.append(content)

        doc.close()

        return results

    # ============================================================
    # PAGE EXTRACTION
    # ============================================================

    def _extract_page(
        self,
        fitz_page,
        plumber_page,
        page_num: int
    ) -> PageContent:

        col_split = self._detect_column_split(plumber_page)

        tables = self._extract_tables(plumber_page, page_num)

        table_bboxes = [
            t.bbox for t in tables if t.bbox
        ]

        text, headings = self._extract_text(
            fitz_page,
            table_bboxes,
            col_split
        )

        images = self._extract_images(
            fitz_page,
            page_num
        )

        return PageContent(
            page_num=page_num,
            text=text,
            tables=tables,
            headings=headings,
            images=images,
        )

    # ============================================================
    # COLUMN DETECTION
    # ============================================================

    def _detect_column_split(self, plumber_page):

        words = plumber_page.extract_words()

        if not words:
            return None

        page_w = plumber_page.width
        mid = page_w / 2

        left_words = sum(
            1 for w in words
            if w["x1"] < mid - 20
        )

        right_words = sum(
            1 for w in words
            if w["x0"] > mid + 20
        )

        if left_words > 10 and right_words > 10:

            ratio = min(left_words, right_words) / max(left_words, right_words)

            if ratio > 0.35:
                return mid

        return None

    # ============================================================
    # TEXT EXTRACTION
    # ============================================================

    def _extract_text(
        self,
        page,
        exclude_bboxes: list,
        col_split: float | None
    ):

        blocks = page.get_text("dict")["blocks"]

        headings = []

        lines_out = []

        for block in blocks:

            if block["type"] != 0:
                continue

            bx0, by0, bx1, by1 = block["bbox"]

            if self._overlaps_any(
                bx0,
                by0,
                bx1,
                by1,
                exclude_bboxes
            ):
                continue

            for line in block["lines"]:

                spans = [
                    s for s in line["spans"]
                    if s["text"].strip()
                ]

                if not spans:
                    continue

                line_text = " ".join(
                    s["text"].strip()
                    for s in spans
                )

                if self._is_noise(line_text):
                    continue

                max_size = max(
                    s.get("size", 0)
                    for s in spans
                )

                is_bold = any(
                    "Bold" in s.get("font", "")
                    or "bold" in s.get("font", "")
                    for s in spans
                )

                is_upper = (
                    line_text.isupper()
                    and len(line_text) > 4
                )

                if (
                    max_size >= self.HEADER_FONT_SIZE
                    and (is_bold or is_upper)
                ):

                    headings.append({
                        "text": line_text,
                        "size": max_size,
                        "bold": is_bold,
                        "page": page.number + 1,
                    })

                lines_out.append(line_text)

        return "\n".join(lines_out), headings

    # ============================================================
    # TABLE VALIDATION
    # ============================================================

    def _is_probable_table(
        self,
        rows: list[list[str]]
    ) -> bool:

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

        max_cols = max(len(r) for r in rows)

        if max_cols < 2:
            return False

        filled = sum(
            1
            for r in rows
            for c in r
            if str(c).strip()
        )

        total = sum(len(r) for r in rows)

        if total == 0:
            return False

        fill_ratio = filled / total

        if fill_ratio < 0.35:
            return False

        col_counts = [len(r) for r in rows]

        if statistics.mean(col_counts) < 2:
            return False

        aligned_rows = 0

        for r in rows:

            non_empty = sum(
                1 for c in r
                if str(c).strip()
            )

            if non_empty >= 2:
                aligned_rows += 1

        if aligned_rows < 2:
            return False

        long_cells = 0

        for r in rows:
            for c in r:

                if len(str(c).split()) > 20:
                    long_cells += 1

        if long_cells > len(rows):
            return False

        avg_words = (
            sum(
                len(str(c).split())
                for r in rows
                for c in r
            )
            / max(total, 1)
        )

        if avg_words < 0.5:
            return False

        return True

    # ============================================================
    # TABLE EXTRACTION
    # ============================================================

    def _extract_tables(
        self,
        plumber_page,
        page_num: int
    ) -> list[TableData]:

        rects = plumber_page.rects

        page_w = plumber_page.width

        h_lines = [
            r for r in rects
            if r["height"] <= 1.5
            and (r["x1"] - r["x0"]) > page_w * 0.15
        ]

        v_lines = [
            r for r in rects
            if r["width"] <= 1.5
            and (r["y1"] - r["y0"]) > 20
        ]

        if len(h_lines) < 2:
            return self._extract_tables_fallback(
                plumber_page,
                page_num
            )

        y_vals = sorted(
            set(round(r["top"]) for r in h_lines)
        )

        clusters = self._cluster_values(
            y_vals,
            gap=45
        )

        tables = []

        for cluster in clusters:

            if len(cluster) < 3:
                continue

            top_y = cluster[0] - 4
            bot_y = cluster[-1] + 8

            cluster_h = [
                r for r in h_lines
                if top_y - 3 <= r["top"] <= bot_y + 3
            ]

            if not cluster_h:
                continue

            x0 = min(r["x0"] for r in cluster_h)
            x1 = max(r["x1"] for r in cluster_h)

            width = x1 - x0
            height = bot_y - top_y

            if width < 120:
                continue

            if height < 40:
                continue

            try:

                crop = plumber_page.crop(
                    (x0, top_y, x1, bot_y)
                )

                words = crop.extract_words(
                    keep_blank_chars=True,
                    use_text_flow=True,
                )

            except Exception:
                continue

            if not words or len(words) < 6:
                continue

            text_area = sum(
                (w["x1"] - w["x0"])
                * (w["bottom"] - w["top"])
                for w in words
            )

            box_area = width * height

            density = text_area / max(box_area, 1)

            if density > 0.55:
                continue

            table_rows = self._words_to_table(
                words,
                cluster,
                x0,
                x1
            )

            if not table_rows:
                continue

            if not self._is_probable_table(table_rows):
                continue

            max_cols = max(
                len(r)
                for r in table_rows
            )

            if max_cols < 2:
                continue

            tables.append(
                TableData(
                    rows=table_rows,
                    page_num=page_num,
                    bbox=(x0, top_y, x1, bot_y),
                )
            )

        if not tables:
            return self._extract_tables_fallback(
                plumber_page,
                page_num
            )

        return tables

    # ============================================================
    # FALLBACK TABLE EXTRACTION
    # ============================================================

    def _extract_tables_fallback(
        self,
        plumber_page,
        page_num: int
    ) -> list[TableData]:

        results = []

        settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "intersection_tolerance": 3,
        }

        try:

            raw_tables = (
                plumber_page.extract_tables(settings)
                or []
            )

            for t in raw_tables:

                if not t:
                    continue

                rows = []

                for r in t:

                    row = [
                        str(c or "").strip()
                        for c in r
                    ]

                    if any(cell for cell in row):
                        rows.append(row)

                if len(rows) < 2:
                    continue

                if not self._is_probable_table(rows):
                    continue

                results.append(
                    TableData(
                        rows=rows,
                        page_num=page_num,
                    )
                )

        except Exception:
            pass

        return results

    # ============================================================
    # WORDS TO TABLE
    # ============================================================

    def _words_to_table(
        self,
        words: list,
        row_y_vals: list,
        x0: float,
        x1: float
    ) -> list[list[str]]:

        if not words:
            return []

        row_intervals = []

        y_sorted = sorted(set(row_y_vals))

        for i in range(len(y_sorted) - 1):

            row_intervals.append(
                (y_sorted[i], y_sorted[i + 1])
            )

        if not row_intervals:
            return []

        all_x0 = sorted(
            set(round(w["x0"]) for w in words)
        )

        col_clusters = self._cluster_values(
            all_x0,
            gap=20
        )

        col_lefts = [
            min(c)
            for c in col_clusters
        ]

        if not col_lefts:
            return []

        num_cols = len(col_lefts)

        rows = {}

        for w in words:

            wy_mid = (
                w["top"] + w["bottom"]
            ) / 2

            row_idx = None

            for ri, (r_top, r_bot) in enumerate(row_intervals):

                if r_top <= wy_mid <= r_bot:
                    row_idx = ri
                    break

            if row_idx is None:

                row_idx = min(
                    range(len(row_intervals)),
                    key=lambda i:
                    abs(
                        (
                            row_intervals[i][0]
                            + row_intervals[i][1]
                        ) / 2 - wy_mid
                    )
                )

            wx = w["x0"]

            col_idx = min(
                range(num_cols),
                key=lambda ci:
                abs(col_lefts[ci] - wx)
            )

            if row_idx not in rows:
                rows[row_idx] = {}

            if col_idx not in rows[row_idx]:
                rows[row_idx][col_idx] = []

            rows[row_idx][col_idx].append(w["text"])

        table = []

        for ri in sorted(rows.keys()):

            row = [
                " ".join(rows[ri].get(ci, []))
                for ci in range(num_cols)
            ]

            if any(c.strip() for c in row):
                table.append(row)

        return table

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _cluster_values(values: list[int], gap: int):

        if not values:
            return []

        clusters = []

        current = [values[0]]

        for v in values[1:]:

            if v - current[-1] <= gap:
                current.append(v)

            else:
                clusters.append(current)
                current = [v]

        clusters.append(current)

        return clusters

    @staticmethod
    def _overlaps_any(
        x0,
        y0,
        x1,
        y1,
        bboxes: list
    ):

        for (bx0, by0, bx1, by1) in bboxes:

            if (
                x0 < bx1
                and x1 > bx0
                and y0 < by1
                and y1 > by0
            ):
                return True

        return False

    def _is_noise(self, text: str):

        for pattern in self.NOISE_PATTERNS:

            if re.search(
                pattern,
                text.strip(),
                re.IGNORECASE
            ):
                return True

        return False

    # ============================================================
    # IMAGE EXTRACTION
    # ============================================================

    @staticmethod
    def _extract_images(
        fitz_page,
        page_num: int
    ) -> list[dict]:

        images = []

        image_list = fitz_page.get_images(full=True)

        for img_index, img in enumerate(image_list):

            xref = img[0]

            rects = fitz_page.get_image_rects(xref)

            if rects:
                rect = rects[0]

                bbox = (
                    rect.x0,
                    rect.y0,
                    rect.x1,
                    rect.y1
                )
            else:
                bbox = None

            images.append({
                "page": page_num,
                "index": img_index,
                "xref": xref,
                "width": img[2],
                "height": img[3],
                "bbox": bbox,
            })

        return images