"""pdf2zh/ocr_prep.py

OCR theo VÙNG cho mọi dạng PDF (tổng quát, không phụ thuộc nội dung/ngành).
Phát triển từ prep_scan.py, dùng chung Tesseract tích hợp trong PyMuPDF như
high_level._ocr_pages (không thêm thư viện mới).

Khác với _ocr_pages cũ (bỏ qua cả trang nếu trang có BẤT KỲ chữ nào, hoặc trang không có ảnh)
và prep_scan.py (chỉ xét trang có ảnh phủ >60% diện tích), mỗi trang được xử lý theo vùng:

  1. Tách chữ trên trang thành: chữ THẬT nhìn thấy (đã dịch được) và lớp chữ OCR VÔ HÌNH
     (render mode 3, kiểu Internet Archive).
  2. Render trang, tô nền lên mọi chữ đã có (cả hai loại). Phần còn lại = vùng thiếu chữ:
     chữ nằm trong ảnh nhỏ, chữ đã thành đường cong, phần trang mà lớp OCR cũ bỏ sót...
  3. Nếu vùng còn lại có mực dạng "chữ" thì OCR riêng vùng đó (Tesseract, ngôn ngữ nguồn).
  4. Gộp dòng từ lớp OCR cũ + dòng OCR mới, lọc nhiễu, rồi như prep_scan: xoá lớp OCR vô hình
     cũ, che chữ trong ảnh bằng ô màu nền lấy mẫu, vẽ lại thành chữ THẬT (font Times).
     Chữ thật đã có, hình vẽ, bảng, công thức: giữ nguyên.

Kết quả: mọi trang đều có chữ thật nhìn thấy được, nên layout model, công thức, bảng và các bản sửa
trong converter.py hoạt động như với PDF bình thường. Ô che nằm trong PDF nên không cần sửa converter.

Biến môi trường (tuỳ chọn):
  PDF2ZH_OCR_DPI        DPI khi render/OCR (mặc định 300)
  PDF2ZH_OCR_MIN_GLYPHS số "mảnh giống chữ" tối thiểu trong vùng thiếu chữ để đáng OCR (mặc định 30)
  PDF2ZH_OCR_FONT       đường dẫn .ttf để vẽ lại chữ (ví dụ C:\\Windows\\Fonts\\times.ttf)

File này cũng chứa đường OCR CŨ (trước đây nằm ở ocr.py), dùng khi PDF2ZH_OCR_MODE=legacy:
  ocr_paragraphs / translate_ocr_page  gom lớp OCR vô hình thành đoạn, dịch thẳng rồi dàn lại
  bằng insert_htmlbox. high_level.py chỉ gọi translate_ocr_page cho các trang mà _ocr_pages trả về;
  chế độ mặc định (region) luôn trả về tập rỗng nên không bao giờ đi đường này.
"""
import logging
import os
from asyncio import CancelledError
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import pymupdf

from pdf2zh.logsetup import set_progress, fmt_pages

logger = logging.getLogger(__name__)

DPI = int(os.environ.get("PDF2ZH_OCR_DPI", "300"))
MIN_GLYPHS = int(os.environ.get("PDF2ZH_OCR_MIN_GLYPHS", "30"))
MIN_PT, MAX_PT = 3.5, 60.0   # cỡ chữ hợp lệ của một dòng OCR (pt)
# Các lớp layout mà pipeline cổ điển KHÔNG dịch (giữ nguyên). Không che/vẽ lại chữ ở đây:
# che phủ lên hình/bảng làm hỏng nội dung mà bản dịch cũng không thay thế.
PROTECTED = ("figure", "table", "isolate_formula", "formula_caption", "abandon")


# --------------------------------------------------------------------------
# Hàm thuần (không cần PyMuPDF để chạy)
# --------------------------------------------------------------------------
def page_bg(arr):
    """Màu nền trang (RGB uint8) = trung vị dải mép 2%."""
    h, w = arr.shape[:2]
    m = max(2, int(0.02 * min(h, w)))
    band = np.concatenate([
        arr[:m].reshape(-1, 3), arr[-m:].reshape(-1, 3),
        arr[:, :m].reshape(-1, 3), arr[:, -m:].reshape(-1, 3),
    ])
    return np.median(band, axis=0).astype(np.uint8)


def has_text_like_ink(masked, bg, scale, min_glyphs=MIN_GLYPHS):
    """Vùng thiếu chữ có đáng OCR không? Đếm các mảnh mực có kích thước cỡ một ký tự.
    Trang trống, nét kẻ, hình khối lớn không tạo ra đủ mảnh nên được bỏ qua (tiết kiệm thời gian)."""
    gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    bg_g = int(0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2])
    ink = (cv2.absdiff(gray, np.full_like(gray, bg_g)) > 60).astype(np.uint8)
    if ink.mean() < 0.0003:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    w = stats[1:, cv2.CC_STAT_WIDTH]
    lo, hi = 2.5 * scale, 18 * scale
    ok = (h >= lo) & (h <= hi) & (w >= 0.6 * scale) & (w <= 2 * hi)
    return int(ok.sum()) >= min_glyphs


def background_and_ink(arr, scale, rect):
    """Màu nền (mode của vùng, lượng tử hoá) và màu chữ tương phản. -> (bg, fg) RGB 0..1."""
    h, w, _ = arr.shape
    x0, y0 = max(int(rect[0] * scale), 0), max(int(rect[1] * scale), 0)
    x1, y1 = min(int(rect[2] * scale) + 1, w), min(int(rect[3] * scale) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return (1.0, 1.0, 1.0), (0, 0, 0)
    region = arr[y0:y1, x0:x1].reshape(-1, 3)
    q = (region >> 4).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    top = Counter(keys.tolist()).most_common(1)[0][0]
    bg = region[keys == top].mean(axis=0) / 255.0
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    if lum > 0.93:
        bg = np.array([1.0, 1.0, 1.0])
    fg = (0, 0, 0) if lum >= 0.5 else (1, 1, 1)
    return tuple(float(c) for c in bg), fg


def keep_line(words, size):
    """Lọc nhiễu OCR: ảnh chụp, hoạ tiết, nét kẻ hay sinh ra dòng 'chữ' toàn ký hiệu / ký tự đơn."""
    if not (MIN_PT <= size <= MAX_PT):
        return False
    core = "".join(words)
    if not core:
        return False
    if sum(c.isalnum() for c in core) < 0.6 * len(core):
        return False
    if sum(c.isalpha() for c in core) < 3:           # dòng toàn số / ký hiệu: khỏi dịch
        return False
    if len(words) > 2 and sum(len(w) == 1 for w in words) > 0.5 * len(words):
        return False
    return True


def _union(rects):
    return (min(r[0] for r in rects), min(r[1] for r in rects),
            max(r[2] for r in rects), max(r[3] for r in rects))


def _intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def lines_from_words(words, size_map, origin_map):
    """words: tuple (x0,y0,x1,y1,text,block,line,wordno). -> [(rects, texts, size, baseline)]."""
    grouped = defaultdict(list)
    for x0, y0, x1, y1, t, b, l, _ in words:
        if t.strip():
            grouped[(b, l)].append(((x0, y0, x1, y1), t))
    lines = []
    for key, items in grouped.items():
        items.sort(key=lambda it: it[0][0])
        rects = [r for r, _ in items]
        h = max(r[3] - r[1] for r in rects)
        size = size_map.get(key) or h * 0.85
        if not 0.5 * h < size < 1.3 * h:             # cỡ chữ vô lý -> ước từ chiều cao
            size = h * 0.85
        base = origin_map.get(key)
        if base is None or not (min(r[1] for r in rects) <= base <= max(r[3] for r in rects) + 1):
            base = max(r[3] for r in rects) - 0.22 * size
        lines.append((rects, [t for _, t in items], size, base))
    return lines


# --------------------------------------------------------------------------
# Phần dùng PyMuPDF
# --------------------------------------------------------------------------
def _size_maps(page):
    sizes, origins = {}, {}
    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES   # không kéo dữ liệu ảnh
    for block in page.get_text("dict", flags=flags)["blocks"]:
        for li, line in enumerate(block.get("lines", [])):
            if line["spans"]:
                sizes[(block["number"], li)] = median(s["size"] for s in line["spans"])
                origins[(block["number"], li)] = line["spans"][0]["origin"][1]
    return sizes, origins


def split_words(page):
    """-> (chữ thật nhìn thấy, chữ thuộc lớp OCR vô hình). Dựa vào render mode của từng span."""
    words = [w for w in page.get_text("words") if w[4].strip()]
    try:
        trace = page.get_texttrace()
    except Exception:
        trace = None
    if trace is None:       # PyMuPDF cũ: dùng quy tắc của prep_scan (ảnh phủ trang + có chữ = lớp OCR)
        area = page.rect.get_area()
        big = any((pymupdf.Rect(i["bbox"]) & page.rect).get_area() > 0.6 * area
                  for i in page.get_image_info())
        return ([], words) if big else (words, [])
    vis = [tuple(pymupdf.Rect(s["bbox"]) + (-1, -1, 1, 1)) for s in trace
           if s.get("type") in (0, 1, 2)]
    visible, invisible = [], []
    for w in words:
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        hit = any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in vis)
        (visible if hit else invisible).append(w)
    return visible, invisible


def protected_boxes(page, model):
    """Khung (toạ độ trang, điểm PDF) của hình/bảng/công thức/đầu-chân trang theo layout model."""
    if model is None:
        return []
    pix = page.get_pixmap(alpha=False, annots=False)        # 72 dpi, giống translate_patch
    image = np.ascontiguousarray(
        np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1])
    res = model.predict(image, imgsz=int(pix.height / 32) * 32)[0]
    out = []
    for b in res.boxes:
        if res.names[int(b.cls)] in PROTECTED:
            x0, y0, x1, y1 = [float(v) for v in np.asarray(b.xyxy).squeeze()]
            out.append((x0, y0, x1, y1))
    return out


def _in_boxes(rects, boxes):
    r = _union(rects)
    cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def _ocr_residual(masked, ctx):
    """OCR ảnh đã che chữ có sẵn. Trả về các dòng (toạ độ đã quy về điểm PDF của trang)."""
    h, w = masked.shape[:2]
    pix = pymupdf.Pixmap(pymupdf.csRGB, w, h, masked.tobytes(), False)
    pix.set_dpi(ctx["dpi"], ctx["dpi"])
    data = pix.pdfocr_tobytes(language=ctx["language"], tessdata=ctx["tessdata"]())
    with pymupdf.open(stream=data) as rec:
        rp = rec[0]
        words = [w for w in rp.get_text("words") if w[4].strip()]
        sizes, origins = _size_maps(rp)
        lines = lines_from_words(words, sizes, origins)
        return lines, (rp.rect.width, rp.rect.height)


def _rescale(lines, src_wh, page):
    fx, fy = page.rect.width / src_wh[0], page.rect.height / src_wh[1]
    if abs(fx - 1) < 1e-3 and abs(fy - 1) < 1e-3:
        return lines
    out = []
    for rects, texts, size, base in lines:
        out.append(([(r[0] * fx, r[1] * fy, r[2] * fx, r[3] * fy) for r in rects],
                    texts, size * fy, base * fy))
    return out


def prep_page(page, ctx):
    if page.rotation:
        return "skip_rotated"
    visible, invisible = split_words(page)
    scale = ctx["dpi"] / 72.0
    pix = page.get_pixmap(dpi=ctx["dpi"], alpha=False, annots=False)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    bg = page_bg(arr)
    masked = arr.copy()
    for w in visible + invisible:                    # che mọi chữ đã có -> chỉ còn vùng thiếu chữ
        x0, y0 = max(int(w[0] * scale) - 2, 0), max(int(w[1] * scale) - 2, 0)
        x1, y1 = int(w[2] * scale) + 3, int(w[3] * scale) + 3
        masked[y0:y1, x0:x1] = bg

    old_lines = []
    if invisible:
        sizes, origins = _size_maps(page)
        old_lines = lines_from_words(invisible, sizes, origins)
    new_lines = []
    if has_text_like_ink(masked, bg, scale):
        lines, wh = _ocr_residual(masked, ctx)
        new_lines = _rescale(lines, wh, page)

    cand = [ln for ln in old_lines + new_lines if keep_line(ln[1], ln[2])]
    if not old_lines and not cand:
        return "native"
    # Dòng nằm trong hình/bảng/công thức: không che, không vẽ lại (pipeline cũng không dịch chúng)
    pbox = protected_boxes(page, ctx["model"]) if cand else []
    lines = [ln for ln in cand if not _in_boxes(ln[0], pbox)]

    # Xoá lớp OCR vô hình cũ (chỉ trong vùng của nó; chữ thật còn nguyên), giữ liên kết
    if old_lines:
        links = page.get_links()
        for rects, _, _, _ in old_lines:
            page.add_redact_annot(pymupdf.Rect(*_union(rects)), fill=False)
        page.apply_redactions(images=0, graphics=0, text=0)
        have = {(round(l["from"].x0), round(l["from"].y0), l.get("uri"), l.get("page"))
                for l in page.get_links()}
        for l in links:
            if (round(l["from"].x0), round(l["from"].y0), l.get("uri"), l.get("page")) not in have:
                page.insert_link(l)

    if not lines:
        return "cleaned_only"
    vis_rects = [tuple(w[:4]) for w in visible]
    font, fname = ctx["font"], ctx["fontname"]
    covers, texts = [], []
    for rects, words, size, base in lines:
        words = [w.replace("\u00ac", "-") for w in words]
        ok = [all(font.has_glyph(ord(c)) for c in w) for w in words]
        runs, cur = [], []
        for rect, word, good in zip(rects, words, ok):
            if good:
                cur.append((rect, word))
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        for run in runs:
            box = _union([r for r, _ in run])
            text = " ".join(w for _, w in run)
            cover = (box[0] - 1.0, box[1] - 2.0, box[2] + 1.0, box[3] + 3.0)
            if any(_intersects(cover, v) for v in vis_rects):   # không đè lên chữ thật bên cạnh
                cover = (box[0] - 0.5, box[1] - 0.5, box[2] + 0.5, box[3] + 0.5)
            bgc, fg = background_and_ink(arr, scale, cover)
            covers.append((cover, bgc))
            natural = font.text_length(text, fontsize=size)
            sx = (box[2] - box[0]) / natural if natural > 0 else 1.0
            sx = min(max(sx, 0.6), 1.5)
            texts.append((pymupdf.Point(box[0], base), text, size, fg, sx))

    for cover, bgc in covers:                         # che hết trước, vẽ chữ sau
        page.draw_rect(pymupdf.Rect(*cover), color=None, fill=bgc, overlay=True)
    for origin, text, size, fg, sx in texts:
        page.insert_text(origin, text, fontname=fname, fontfile=ctx["fontfile"],
                         fontsize=size, color=fg,
                         morph=(origin, pymupdf.Matrix(sx, 0, 0, 1, 0, 0)), overlay=True)
    return "converted_old" if old_lines and not new_lines else \
        ("converted_new" if new_lines and not old_lines else "converted_both")


def prep_document(doc, pages, language, tessdata_fn, cancellation_event=None, model=None):
    """Xử lý từng trang của `doc` (bản làm việc). Trả về tập số trang đi đường OCR cũ: luôn rỗng,
    vì trang đã có chữ thật nên đi tiếp pipeline cổ điển như PDF thường."""
    fontfile = os.environ.get("PDF2ZH_OCR_FONT")
    fontfile = fontfile if fontfile and Path(fontfile).exists() else None
    if fontfile:
        font, fname = pymupdf.Font(fontfile=fontfile), "ocrprep"
    else:
        font, fname = pymupdf.Font("tiro"), "tiro"
    memo = {}

    def tessdata():
        if "v" not in memo:
            memo["v"] = tessdata_fn()
        return memo["v"]

    ctx = dict(dpi=DPI, language=language, tessdata=tessdata, model=model,
               font=font, fontname=fname, fontfile=fontfile)
    stats = Counter()
    failed = []
    for pageno in range(len(doc)):
        if cancellation_event and cancellation_event.is_set():
            raise CancelledError("task cancelled")
        if pages and pageno not in pages:
            continue
        set_progress(f"OCR prep page {pageno + 1}/{len(doc)}")
        try:
            kind = prep_page(doc[pageno], ctx)
        except CancelledError:
            raise
        except Exception as exc:                     # một trang lỗi không làm hỏng cả file
            kind = "error"
            failed.append(pageno)
            logger.warning("OCR prep page %d failed: %s: %s", pageno + 1,
                           type(exc).__name__, exc, exc_info=True)
        stats[kind] += 1
        if kind.startswith("converted"):
            logger.debug("OCR prep page %d: %s", pageno + 1, kind)   # từng trang: chỉ DEBUG, tổng kết ở INFO bên dưới
    logger.info("OCR prep (%s): %s", language, dict(stats))
    if failed:
        logger.warning("OCR prep failed on %s", fmt_pages(failed))
    return set()


# --------------------------------------------------------------------------
# Đường OCR CŨ (PDF2ZH_OCR_MODE=legacy) - gộp từ ocr.py
# --------------------------------------------------------------------------

def ocr_paragraphs(page, detection):
    """Group OCR words by layout region, then join lines in reading order."""
    regions = [
        (pymupdf.Rect(*box.xyxy), detection.names[int(box.cls)], box.conf)
        for box in detection.boxes
    ]
    line_sizes = {
        (block["number"], i): median(span["size"] for span in line["spans"])
        for block in page.get_text(
            "dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
        )["blocks"]
        for i, line in enumerate(block.get("lines", []))
        if line["spans"]
    }
    groups = defaultdict(list)
    protected = {"figure", "table", "isolate_formula", "formula_caption"}
    for word in page.get_text("words"):
        rect = pymupdf.Rect(word[:4]) * page.rotation_matrix
        if rect.is_empty or not word[4].strip():
            continue
        center = (rect.tl + rect.br) / 2
        if any(center in box and name in protected for box, name, _ in regions):
            continue
        matches = [
            ((pymupdf.Rect(rect) & box).get_area(), confidence, i)
            for i, (box, _, confidence) in enumerate(regions)
        ]
        overlap, _, region = max(matches, default=(0, 0, -1))
        if overlap and regions[region][1] == "abandon":
            continue
        key = region if overlap else ("unassigned", word[5])
        groups[key].append((rect, word[4], word[5:7]))

    paragraphs = []
    for words in groups.values():
        line_height = median(w[0].height for w in words)
        ordered = []
        for word in sorted(words, key=lambda w: (w[0].y0 + w[0].y1, w[0].x0)):
            center = (word[0].y0 + word[0].y1) / 2
            if (
                not ordered
                or abs(center - median((w[0].y0 + w[0].y1) / 2 for w in ordered[-1]))
                > line_height * 0.5
            ):
                ordered.append([])
            ordered[-1].append(word)
        left = min(w[0].x0 for w in words)
        chunks = []
        for line in ordered:
            line.sort(key=lambda w: w[0].x0)
            bounds = pymupdf.Rect(line[0][0])
            for word in line[1:]:
                bounds |= word[0]
            if chunks:
                previous = chunks[-1][-1][0]
                if (
                    bounds.y0 - previous.y1 > line_height * 0.8
                    or bounds.x0 > left + line_height * 0.8
                    or bounds.x0 >= previous.x1
                    or bounds.x1 <= previous.x0
                ):
                    chunks.append([])
            else:
                chunks.append([])
            chunks[-1].append((bounds, " ".join(w[1] for w in line), line))
        for chunk in chunks:
            bounds = pymupdf.Rect(chunk[0][0])
            text = ""
            source = []
            sizes = []
            for rect, line, words in chunk:
                bounds |= rect
                # ponytail: a lowercase continuation treats an end-of-line hyphen as soft.
                if text.endswith("-") and line[:1].islower():
                    text = text[:-1] + line
                else:
                    text += (" " if text else "") + line
                source.extend(w[0] for w in words)
                sizes.extend(
                    line_sizes.get(tuple(w[2]), w[0].height * 0.85) for w in words
                )
            size = median(sizes)
            bounds.y1 = min(page.rect.y1, max(bounds.y1, bounds.y0 + size * 1.25))
            paragraphs.append((bounds, text, size, source))
    # OCR glyph boxes can overlap the next paragraph by a few points.
    for i, (above, _, _, _) in enumerate(paragraphs):
        for below, _, _, _ in paragraphs[i + 1 :]:
            a, b = sorted((above, below), key=lambda r: r.y0)
            overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
            if a.y0 < b.y0 < a.y1 and overlap > min(a.width, b.width) / 2:
                boundary = (a.y1 + b.y0) / 2
                a.y1, b.y0 = boundary - 0.5, boundary + 0.5
    return paragraphs


def translate_ocr_page(
    page, detection, translator, font, thread, cancellation_event=None
):
    paragraphs = ocr_paragraphs(page, detection)
    if not paragraphs:
        return

    from tenacity import retry, wait_fixed  # chỉ cần ở chế độ legacy

    @retry(wait=wait_fixed(1))
    def translate(paragraph):
        if cancellation_event and cancellation_event.is_set():
            raise CancelledError("task cancelled")
        return translator.translate(paragraph[1])

    with ThreadPoolExecutor(max_workers=max(1, thread)) as executor:
        translations = list(executor.map(translate, paragraphs))
    if not any(text.strip() for text in translations):
        return
    archive = pymupdf.Archive((font.buffer, "ocr.ttf"))
    with pymupdf.Document() as overlay:
        target = overlay.new_page(width=page.rect.width, height=page.rect.height)
        for (_, _, _, words), text in zip(paragraphs, translations):
            if text.strip():
                for rect in words:
                    target.draw_rect(
                        rect + (-0.5, -0.5, 0.5, 0.5), color=None, fill=(1, 1, 1)
                    )
        for (bounds, _, size, _), text in zip(paragraphs, translations):
            if not text.strip():
                continue
            centered = any(
                detection.names[int(box.cls)] == "title"
                and (pymupdf.Rect(*box.xyxy) & bounds).get_area() > bounds.get_area() / 2
                for box in detection.boxes
            )
            align = "center" if centered else "justify"
            css = (
                "@font-face {font-family:ocr;src:url(ocr.ttf);}"
                f"body {{font-family:ocr;font-size:{size}pt;line-height:1.2;margin:0;}}"
                f"p {{margin:0;overflow-wrap:break-word;text-align:{align};}}"
            )
            spare, _ = target.insert_htmlbox(
                bounds, f"<p>{escape(text)}</p>", css=css, archive=archive, scale_low=0
            )
            if spare < 0:
                raise RuntimeError("OCR translation could not fit its paragraph box")
        # Remove the invisible OCR layer, keeping source pixels and annotations.
        links = page.get_links()
        page.add_redact_annot(page.rect * page.derotation_matrix, fill=False)
        page.apply_redactions(images=0, graphics=0, text=0)
        for link in links:
            page.insert_link(link)
        page.show_pdf_page(
            page.rect * page.derotation_matrix, overlay, 0, rotate=page.rotation
        )