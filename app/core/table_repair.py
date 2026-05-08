"""
app/core/table_repair.py
Post-processing untuk memperbaiki tabel hasil OCR scan yang tidak sempurna.
"""

import re
from difflib import get_close_matches


KNOWN_JABATAN = [
    "Ketua Penguji", "Penguji Utama", "Penguji I", "Penguji II", "Penguji III",
    "Sekretaris", "Anggota I", "Anggota II", "Anggota III", "Anggota IV", "Anggota V",
    "Pembimbing I", "Pembimbing II", "Pembimbing Utama", "Pembimbing Pendamping",
    "Ketua", "Wakil Ketua", "Jabatan",
]

# Regex: string yang HANYA berisi gelar/suffix akademik (bukan nama orang)
_SUFFIX_ONLY = re.compile(
    r"^(S\.E|S\.H|S\.T|M\.Si|M\.SI|M\.S|M\.Pd|M\.M|M\.Kom|M\.Ag|"
    r"Ph\.D|Dr|Drs|Prof|S\.Sos|S\.Pd|S\.Kom|S\.Ag|M\.Hum|M\.P|"
    r"[A-Z]\.[A-Za-z]+)([\.,\s]|$)"
)


def _clean_cell(val, preserve_newlines: bool = False) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ("nan", "none", ""):
        return ""
    if not preserve_newlines:
        s = re.sub(r'\n+', ' ', s)
    s = re.sub(r' {2,}', ' ', s)
    return s.strip()


def _is_noise_cell(val: str) -> bool:
    if not val:
        return True
    alpha_ratio = sum(c.isalpha() for c in val) / max(len(val), 1)
    if alpha_ratio < 0.3 and len(val) <= 8:
        return True
    return False


def _is_noise_row(row: list) -> bool:
    meaningful = [c for c in row if c and not _is_noise_cell(str(c))]
    return len(meaningful) == 0


def _fuzzy_jabatan(text: str) -> str:
    if not text:
        return text
    text_lower = text.lower().strip()
    for j in KNOWN_JABATAN:
        if j.lower() == text_lower:
            return j
    prefix_fixes = {
        "nggota": "Anggota", "enguji": "Penguji",
        "etua": "Ketua", "ekretaris": "Sekretaris", "embimbing": "Pembimbing",
    }
    for fragment, replacement in prefix_fixes.items():
        if text_lower.startswith(fragment.lower()):
            fixed = replacement + text[len(fragment):]
            matches = get_close_matches(fixed, KNOWN_JABATAN, n=1, cutoff=0.7)
            if matches:
                return matches[0]
    matches = get_close_matches(text, KNOWN_JABATAN, n=1, cutoff=0.6)
    return matches[0] if matches else text


def _split_merged_names(cell: str) -> list:
    """
    Split sel yang berisi multiple nama (dipisah newline).
    Jika bagian pertama hanya suffix gelar → itu sisa dari nama baris sebelumnya,
    kembalikan hanya nama dari bagian kedua dan seterusnya.
    """
    if not cell or '\n' not in cell:
        return [cell] if cell else []

    parts = [p.strip() for p in cell.split('\n') if p.strip()]
    if len(parts) < 2:
        return [cell]

    # Jika bagian pertama hanya suffix gelar (S.E, M.SI, dll) → sisa dari nama sebelumnya
    first = parts[0]
    if _SUFFIX_ONLY.match(first) and len(first) < 30:
        # Ambil bagian kedua dan seterusnya sebagai nama lengkap
        remaining = [p for p in parts[1:] if p and re.search(r'[A-Z]', p)]
        if remaining:
            return remaining
        return [cell]

    # Semua bagian terlihat seperti nama → split
    valid = [p for p in parts if re.search(r'[A-Z]', p) and len(p) > 3]
    return valid if len(valid) > 1 else [cell]


def is_penguji_table(rows: list) -> bool:
    if not rows:
        return False
    header = [str(c).lower().strip() for c in rows[0]]
    header_joined = " ".join(header)
    if any(kw in header_joined for kw in ["jabatan", "tanda tangan"]):
        return True
    jabatan_count = 0
    for row in rows[1:]:
        if row and row[0]:
            if get_close_matches(str(row[0]), KNOWN_JABATAN, n=1, cutoff=0.55):
                jabatan_count += 1
    return jabatan_count >= 2


def repair_table(rows: list) -> list:
    """Perbaiki tabel OCR. Auto-detect tipe tabel dan terapkan repair."""
    if not rows:
        return rows
    # Preserve newlines di kolom nama untuk deteksi multiple nama
    normalized = [[_clean_cell(c, preserve_newlines=(ci == 1)) 
                    for ci, c in enumerate(row)] for row in rows]
    filtered = [r for r in normalized if not _is_noise_row(r)]
    if not filtered:
        return rows
    if is_penguji_table(filtered):
        return _repair_penguji_table(filtered)
    return _repair_generic(filtered)


def _repair_penguji_table(rows: list) -> list:
    """Repair khusus tabel penguji/pembimbing."""
    result = []
    num_cols = max(len(r) for r in rows)
    if num_cols < 3:
        rows = [r + [""] * (3 - len(r)) for r in rows]
        num_cols = 3
    rows = [list(r) + [""] * (num_cols - len(r)) for r in rows]

    # Deteksi dan normalisasi header
    first_row_lower = [c.lower() for c in rows[0]]
    has_header = any("jabatan" in c or "nama" in c for c in first_row_lower)
    if has_header:
        header = []
        for c in rows[0]:
            cl = c.lower().strip()
            if "jabatan" in cl:
                header.append("Jabatan")
            elif "nama" in cl:
                header.append("Nama")
            elif "tanda" in cl or "tangan" in cl:
                header.append("Tanda Tangan")
            else:
                header.append(c or "")
        result.append(header)
        data_rows = rows[1:]
    else:
        result.append(["Jabatan", "Nama", "Tanda Tangan"] + [""] * (num_cols - 3))
        data_rows = rows

    expected_order = [
        "Ketua Penguji", "Penguji Utama", "Sekretaris",
        "Anggota I", "Anggota II", "Anggota III", "Anggota IV",
    ]
    jabatan_idx = 0
    i = 0

    while i < len(data_rows):
        row = data_rows[i]
        jabatan_raw = row[0] if row else ""
        nama_raw = row[1] if len(row) > 1 else ""

        # Fix jabatan kosong → inferensi dari urutan
        if not jabatan_raw and jabatan_idx < len(expected_order):
            jabatan_fixed = expected_order[jabatan_idx]
        else:
            jabatan_fixed = _fuzzy_jabatan(jabatan_raw) if jabatan_raw else ""

        # Update pointer urutan
        if jabatan_fixed in expected_order:
            try:
                jabatan_idx = expected_order.index(jabatan_fixed) + 1
            except ValueError:
                jabatan_idx += 1

        # Cek nama: ada multiple nama yang terpecah?
        nama_parts = _split_merged_names(nama_raw)

        if len(nama_parts) > 1:
            # Baris pertama
            result.append([jabatan_fixed, nama_parts[0]] + [""] * (num_cols - 2))
            # Baris berikutnya dengan jabatan dari urutan
            for part_nama in nama_parts[1:]:
                next_j = expected_order[jabatan_idx] if jabatan_idx < len(expected_order) else ""
                jabatan_idx += 1
                result.append([next_j, part_nama] + [""] * (num_cols - 2))
        else:
            nama_clean = nama_parts[0] if nama_parts else ""
            # Skip baris yang hanya noise dan tidak ada jabatan
            if not jabatan_fixed and _is_noise_cell(nama_clean):
                i += 1
                continue
            if jabatan_fixed or nama_clean:
                result.append([jabatan_fixed, nama_clean] + [""] * (num_cols - 2))

        i += 1

    # Deduplicate
    seen = set()
    deduped = []
    for row in result:
        key = tuple(str(c) for c in row)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def _repair_generic(rows: list) -> list:
    if not rows:
        return rows
    num_cols = max(len(r) for r in rows)
    result = []
    for row in rows:
        row = list(row) + [""] * (num_cols - len(row))
        row = row[:num_cols]
        cleaned = [_clean_cell(c) for c in row]
        if not _is_noise_row(cleaned):
            result.append(cleaned)
    return result