"""
app/core/table_extractor.py
Microsoft Table Transformer untuk mendeteksi dan mengekstrak tabel
"""

import fitz
from PIL import Image
import torch
import logging
from transformers import TableTransformerForObjectDetection, DetrImageProcessor

logger = logging.getLogger(__name__)

class TableExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[Table Transformer] Menggunakan device: {self.device}")

        self.processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        ).to(self.device)

    def extract_tables_from_pdf(self, pdf_path: str, page_num: int, dpi: int = 300):
        """Ekstrak semua tabel dari satu halaman PDF"""
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]

        # Render halaman ke gambar berkualitas tinggi
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()

        # Deteksi tabel
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.65
        )[0]

        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label == 0:  # 0 = table class
                x1, y1, x2, y2 = map(int, box.tolist())
                cropped = image.crop((x1, y1, x2, y2))

                tables.append({
                    "page_num": page_num,
                    "bbox": (x1, y1, x2, y2),
                    "image": cropped,
                    "confidence": float(score)
                })

        return tables