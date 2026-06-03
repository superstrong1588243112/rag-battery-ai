import io
import os
import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None

try:
    from config import Config
except Exception:
    Config = None


SENTENCE_ENDINGS = tuple("。！？；.!?;")
NOISE_PATTERNS = [
    r"^万方数据$",
    r"^www\..*",
    r"^\d+\s*$",
    r"^第\s*\d+\s*[卷期].*",
    r"^20\d{2}\s*年\s*第\s*\d+\s*卷.*",
    r"^储能科学与技术\s*$",
    r"^Energy Storage Science and Technology$",
    r"^doi[:：\s]",
]


def _is_cjk(char):
    return "\u4e00" <= char <= "\u9fff"


def _normalize_inline(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([，。！？；：、,.!?;:%）\]\}])", r"\1", text)
    text = re.sub(r"([（\[\{])\s+", r"\1", text)
    text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", text)
    text = re.sub(r"\bLi\s+([A-Z][A-Za-z]?)\s*([0-9])\b", r"Li\1\2", text)
    text = re.sub(r"\b([A-Z][a-z]?)\s+([0-9])\b", r"\1\2", text)
    return text.strip()


def looks_like_noise(line):
    line = _normalize_inline(line)
    if not line:
        return True
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, line, flags=re.IGNORECASE):
            return True
    if len(line) <= 2 and not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", line):
        return True
    return False


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c for c in text if ord(c) >= 32 or c in "\n\t")
    lines = []
    previous = None

    for raw in text.splitlines():
        line = _normalize_inline(raw)
        if looks_like_noise(line):
            continue
        if line == previous:
            continue
        previous = line
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"([一二三四五六七八九十]+)\s*[、.]\s*", r"\1、", text)
    return text.strip()


def _extract_lines_from_page(page):
    raw = page.get_text("dict")
    lines = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans)
            text = _normalize_inline(text)
            if not text or looks_like_noise(text):
                continue
            x0, y0, x1, y1 = line.get("bbox", block.get("bbox", (0, 0, 0, 0)))
            lines.append({
                "text": text,
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "width": float(x1 - x0),
            })
    return lines


def detect_column_boundary(lines, page_width):
    centers = sorted((line["x0"] + line["x1"]) / 2 for line in lines)
    if len(centers) < 8:
        return page_width / 2

    gaps = []
    for left, right in zip(centers, centers[1:]):
        gap = right - left
        mid = (left + right) / 2
        if page_width * 0.35 < mid < page_width * 0.65:
            gaps.append((gap, mid))

    if not gaps:
        return page_width / 2

    gap, mid = max(gaps, key=lambda item: item[0])
    return mid if gap > page_width * 0.06 else page_width / 2


def is_two_column_layout_fitz(page):
    lines = _extract_lines_from_page(page)
    if len(lines) < 12:
        return False

    width = page.rect.width
    height = page.rect.height
    candidates = [
        line for line in lines
        if width * 0.08 < line["x0"] < width * 0.92
        and height * 0.08 < line["y0"] < height * 0.94
        and line["width"] < width * 0.58
    ]
    if len(candidates) < 10:
        return False

    left = [line for line in candidates if (line["x0"] + line["x1"]) / 2 < width * 0.48]
    right = [line for line in candidates if (line["x0"] + line["x1"]) / 2 > width * 0.52]
    if len(left) < 4 or len(right) < 4:
        return False

    left_y = (min(line["y0"] for line in left), max(line["y1"] for line in left))
    right_y = (min(line["y0"] for line in right), max(line["y1"] for line in right))
    overlap = min(left_y[1], right_y[1]) - max(left_y[0], right_y[0])
    return overlap > height * 0.25


def _glue_text(left, right):
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-"):
        return left[:-1] + right
    if _is_cjk(left[-1]) and _is_cjk(right[0]):
        return left + right
    if left[-1].isalnum() and right[0].isalnum():
        return left + " " + right
    return left + right


def _lines_to_paragraphs(lines):
    if not lines:
        return ""
    lines = sorted(lines, key=lambda item: (item["y0"], item["x0"]))
    paragraphs = []
    current = ""
    previous = None

    for line in lines:
        text = line["text"]
        if previous is None:
            current = text
            previous = line
            continue

        vertical_gap = line["y0"] - previous["y1"]
        starts_section = bool(re.match(r"^(\d+(\.\d+)*|[一二三四五六七八九十]+)[、.]\s*", text))
        if current.endswith(SENTENCE_ENDINGS) or vertical_gap > 12 or starts_section:
            paragraphs.append(current.strip())
            current = text
        else:
            current = _glue_text(current, text)
        previous = line

    if current:
        paragraphs.append(current.strip())

    return "\n\n".join(p for p in paragraphs if p)


def extract_text_by_columns_fitz(page):
    lines = _extract_lines_from_page(page)
    if not lines:
        return page.get_text("text") or ""

    width = page.rect.width
    boundary = detect_column_boundary(lines, width)
    gutter = max(18, width * 0.035)
    two_column = is_two_column_layout_fitz(page)

    if not two_column:
        return _lines_to_paragraphs(lines)

    left, right, full = [], [], []
    for line in lines:
        center = (line["x0"] + line["x1"]) / 2
        crosses_gutter = line["x0"] < boundary - gutter and line["x1"] > boundary + gutter
        if crosses_gutter or line["width"] > width * 0.68:
            full.append(line)
        elif center < boundary:
            left.append(line)
        else:
            right.append(line)

    column_lines = left + right
    if column_lines:
        min_col_y = min(line["y0"] for line in column_lines)
        max_col_y = max(line["y1"] for line in column_lines)
    else:
        min_col_y = max_col_y = 0

    top_full = [line for line in full if line["y1"] < min_col_y + 8]
    bottom_full = [line for line in full if line["y0"] > max_col_y - 8]
    middle_full = [line for line in full if line not in top_full and line not in bottom_full]

    parts = [
        _lines_to_paragraphs(top_full),
        _lines_to_paragraphs(left),
        _lines_to_paragraphs(right),
        _lines_to_paragraphs(middle_full + bottom_full),
    ]
    return "\n\n".join(part for part in parts if part)


def _page_to_ocr_text(page):
    if Image is None or pytesseract is None:
        return ""
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    try:
        return pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
    except Exception:
        return ""


def _tables_to_markdown(tables):
    blocks = []
    for idx, table in enumerate(tables or [], start=1):
        rows = []
        for row in table:
            cells = [clean_text(str(cell or "")) for cell in row]
            if any(cells):
                rows.append(cells)
        if not rows:
            continue
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        header = rows[0]
        block = [f"[表格 {idx}]", "| " + " | ".join(header) + " |"]
        block.append("| " + " | ".join(["---"] * width) + " |")
        for row in rows[1:]:
            block.append("| " + " | ".join(row) + " |")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def load_pdf_pages(file_path, max_pages=None):
    documents = []
    path = Path(file_path)
    table_pages = {}

    if Config is None or getattr(Config, "EXTRACT_TABLES", True):
        if pdfplumber is not None:
            try:
                with pdfplumber.open(str(path)) as pdf:
                    for idx, page in enumerate(pdf.pages):
                        if max_pages is not None and idx >= max_pages:
                            break
                        table_pages[idx] = _tables_to_markdown(page.extract_tables())
            except Exception:
                table_pages = {}

    with fitz.open(str(path)) as pdf:
        total = len(pdf)
        for page_index, page in enumerate(pdf):
            if max_pages is not None and page_index >= max_pages:
                break
            text = extract_text_by_columns_fitz(page)
            text = clean_text(text)

            if len(text) < 80 and Config is not None and getattr(Config, "OCR_IF_NO_TEXT", False):
                ocr_text = clean_text(_page_to_ocr_text(page))
                if len(ocr_text) > len(text):
                    text = ocr_text

            table_text = table_pages.get(page_index, "")
            if table_text:
                text = f"{text}\n\n{table_text}".strip()

            if not text:
                continue

            metadata = {
                "source": path.name,
                "file_path": str(path),
                "file_type": "pdf",
                "page": page_index + 1,
                "total_pages": total,
                "category": classify_document(path.name),
                "topic": extract_topic(path.name),
            }
            documents.append(Document(page_content=text, metadata=metadata))

    return documents


def load_pdf(file_path):
    pages = load_pdf_pages(file_path)
    return "\n\n".join(page.page_content for page in pages)


def load_documents(folder_path):
    documents = []
    folder = Path(folder_path)
    for file_path in sorted(folder.glob("*.pdf")):
        try:
            pages = load_pdf_pages(file_path)
            documents.extend(pages)
            print(f"加载成功: {file_path.name} ({len(pages)} 页)")
        except Exception as exc:
            print(f"加载失败: {file_path.name}: {exc}")
    return documents


def classify_document(filename):
    if "电化学基础" in filename:
        return "电化学基础"
    if "基础科学问题" in filename:
        return "锂电池基础科学"
    if "失效分析" in filename:
        return "失效分析"
    return "其他"


def extract_topic(filename):
    topic_keywords = {
        "法拉第": "法拉第定律",
        "能斯特": "能斯特方程",
        "电极过程": "电极过程动力学",
        "能量密度": "能量密度",
        "缺陷": "材料缺陷",
        "相图": "相图与相变",
        "界面": "电池界面",
        "输运": "离子输运",
        "正极": "正极材料",
        "负极": "负极材料",
        "电解质": "电解质",
        "全固态": "全固态电池",
        "锂空气": "锂空气电池",
        "锂硫": "锂硫电池",
        "表征": "表征方法",
        "测量": "电化学测量",
        "计算": "计算方法",
        "失效": "失效分析",
    }
    for keyword, topic in topic_keywords.items():
        if keyword in filename:
            return topic
    return "综合"


def split_documents(documents, chunk_size=900, chunk_overlap=160):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        source = chunk.metadata.get("source", "未知来源")
        page = chunk.metadata.get("page")
        chunk.metadata["source_ref"] = f"{source} 第{page}页" if page else source
    return chunks


# Backward-compatible names used by the old test scripts.
def is_two_column_layout(page):
    return False


def detect_column_boundaries(page):
    return page.width / 2


def extract_text_by_columns(page):
    return page.extract_text() or ""
