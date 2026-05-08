"""
app/core/markdown.py

Generate markdown:
- TOC
- Tables
- Images
- Formulas
- Image captions
"""

import os
import re

from .chunker import DocumentChunk


class MarkdownGenerator:
    def generate(
        self,
        chunks: list[DocumentChunk],
        doc_title: str = "Dokumen"
    ) -> str:

        parts = []

        parts.append(f"# {doc_title}\n")

        parts.append("---\n")

        toc = self._build_toc(chunks)

        if toc:

            parts.append("## Daftar Isi\n")

            parts.append(toc)

            parts.append("\n---\n")

        for chunk in chunks:

            parts.append(
                self._render_chunk(chunk)
            )

        return "\n".join(parts)

    def _build_toc(
        self,
        chunks: list[DocumentChunk]
    ) -> str:

        lines = []

        seen = set()

        for chunk in chunks:

            title = chunk.title.strip()

            if (
                not title
                or title in seen
                or len(title) < 3
            ):
                continue

            seen.add(title)

            indent = "  " * (chunk.level - 1)

            anchor = self._to_anchor(title)

            lines.append(
                f"{indent}- [{title}](#{anchor})"
            )

        return "\n".join(lines)

    def _render_chunk(
        self,
        chunk: DocumentChunk
    ) -> str:

        parts = []

        heading_level = min(chunk.level + 1, 4)

        heading = "#" * heading_level

        parts.append(
            f"\n{heading} {chunk.title}"
        )

        if chunk.page_start == chunk.page_end:

            parts.append(
                f"*Halaman {chunk.page_start}*\n"
            )

        else:

            parts.append(
                f"*Halaman {chunk.page_start}–{chunk.page_end}*\n"
            )

        if chunk.content.strip():

            cleaned = self._clean_content(
                chunk.content
            )

            parts.append(cleaned)

        if chunk.formulas:

            parts.append("\n## Rumus\n")

            for formula in chunk.formulas:

                formula_text = formula.get(
                    "text",
                    ""
                ).strip()

                if formula_text:

                    parts.append(
                        f"```math\n{formula_text}\n```\n"
                    )

        for i, table in enumerate(chunk.tables):

            md_table = self._render_table(
                table,
                i + 1,
                chunk
            )

            if md_table:
                parts.append(md_table)

        if chunk.images:

            parts.append("\n## Gambar\n")

            for img in chunk.images:

                path = img.get("path", "")

                caption = img.get("caption", "")

                filename = os.path.basename(path)

                relative = f"../images/{filename}"

                parts.append(
                    f"![{caption}]({relative})"
                )

                if caption:
                    parts.append(
                        f"\n*{caption}*\n"
                    )

        parts.append("\n---")

        return "\n".join(parts)

    def _render_table(
        self,
        table,
        idx: int,
        chunk: DocumentChunk
    ) -> str:

        if hasattr(table, "rows"):

            rows = table.rows

            caption = getattr(
                table,
                "caption",
                ""
            )

            page_num = getattr(
                table,
                "page_num",
                chunk.page_start
            )

        elif isinstance(table, list):

            rows = table

            caption = ""

            page_num = chunk.page_start

        else:
            return ""

        if not rows or len(rows) < 2:
            return ""

        rows = [
            r for r in rows
            if any(str(c or "").strip() for c in r)
        ]

        if len(rows) < 2:
            return ""

        lines = []

        if caption:

            lines.append(
                f"\n**Tabel {idx}: {caption}** *(Halaman {page_num})*\n"
            )

        else:

            lines.append(
                f"\n**Tabel {idx}** *(Halaman {page_num})*\n"
            )

        max_cols = max(len(r) for r in rows)

        normalized = []

        for row in rows:

            cells = [
                str(c or "").strip().replace("\n", " ").replace("|", "\\|")
                for c in row
            ]

            while len(cells) < max_cols:
                cells.append("")

            normalized.append(cells[:max_cols])

        # header
        lines.append(
            "| " + " | ".join(normalized[0]) + " |"
        )

        lines.append(
            "| " + " | ".join(["---"] * max_cols) + " |"
        )

        # body
        for row in normalized[1:]:

            lines.append(
                "| " + " | ".join(row) + " |"
            )

        lines.append("")

        return "\n".join(lines)

    def _clean_content(
        self,
        text: str
    ) -> str:

        lines = text.split("\n")

        paragraphs = []

        current = []

        for line in lines:

            line = line.strip()

            if not line:

                if current:

                    paragraphs.append(
                        " ".join(current)
                    )

                    current = []

            else:

                current.append(line)

        if current:

            paragraphs.append(
                " ".join(current)
            )

        return "\n\n".join(
            p for p in paragraphs if p
        )

    @staticmethod
    def _to_anchor(text: str):

        anchor = text.lower().strip()

        anchor = re.sub(
            r"[^\w\s-]",
            "",
            anchor
        )

        anchor = re.sub(
            r"[\s_]+",
            "-",
            anchor
        )

        return anchor.strip("-")