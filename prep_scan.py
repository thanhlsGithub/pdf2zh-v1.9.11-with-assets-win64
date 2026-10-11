#!/usr/bin/env python3
"""
prep_scan.py - Biến PDF scan thành PDF "chữ thật" để pdf2zh / BabelDOC dịch được.

Với mỗi trang scan:
  1. Lấy chữ + vị trí: dùng lớp OCR có sẵn (kiểu Internet Archive) hoặc tự OCR
     bằng Tesseract tích hợp trong PyMuPDF (không cần cài Tesseract riêng).
  2. Xóa lớp chữ OCR vô hình cũ (nếu có).
  3. Che chữ trong ảnh nền bằng ô màu nền (trắng, hoặc màu nền lấy mẫu từ ảnh).
  4. Vẽ lại chữ đó thành chữ THẬT, nhìn thấy được (font Times).
Hình vẽ, biểu đồ, bảng nằm ngoài vùng chữ được giữ nguyên.

Trang đã có chữ gốc (PDF digital) được giữ nguyên, không đụng tới.

Cách dùng:
  python prep_scan.py input.pdf output.pdf
  python prep_scan.py input.pdf output.pdf --pages 1-20         # thử vài trang trước
  python prep_scan.py input.pdf output.pdf --workers 4          # chạy song song
  python prep_scan.py input.pdf output.pdf --lang eng --font C:/Windows/Fonts/times.ttf

Yêu cầu: pip install pymupdf numpy
"""
import argparse
import os
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median

import numpy as np
import pymupdf

TESSDATA_URL = (
    "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/{}.traineddata"
)


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def get_tessdata(lang):
    """Trả về thư mục tessdata, tự tải ngôn ngữ cần thiết lần đầu."""
    if os.environ.get("TESSDATA_PREFIX"):
        return os.environ["TESSDATA_PREFIX"]
    cache = Path.home() / ".cache" / "prep_scan" / "tessdata"
    cache.mkdir(parents=True, exist_ok=True)
    for code in lang.split("+"):
        f = cache / f"{code}.traineddata"
        if not f.exists():
            print(f"Đang tải dữ liệu OCR '{code}' ...", flush=True)
            urllib.request.urlretrieve(TESSDATA_URL.format(code), f)
    return str(cache)


def parse_pages(spec, total):
    if not spec:
        return list(range(total))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a) - 1, min(int(b), total))
        else:
            out.append(int(part) - 1)
    return sorted({p for p in out if 0 <= p < total})


def page_kind(page):
    """'native' | 'scan_with_text' | 'scan' | 'empty'."""
    has_text = bool(page.get_text().strip())
    area = page.rect.get_area()
    big_img = any(
        (pymupdf.Rect(i["bbox"]) & page.rect).get_area() > 0.6 * area
        for i in page.get_image_info()
    )
    if big_img and has_text:
        return "scan_with_text"  # lớp OCR vô hình đè lên ảnh trang
    if big_img:
        return "scan"
    return "native" if has_text else "empty"


def extract_lines(src_page):
    """Gom từ theo dòng: [(danh sách rect từ, danh sách chữ, cỡ chữ, baseline_y)]."""
    words = defaultdict(list)
    for x0, y0, x1, y1, w, b, l, _ in src_page.get_text("words"):
        if w.strip():
            words[(b, l)].append((pymupdf.Rect(x0, y0, x1, y1), w))
    sizes, origins = {}, {}
    for block in src_page.get_text("dict")["blocks"]:
        for li, line in enumerate(block.get("lines", [])):
            if line["spans"]:
                sizes[(block["number"], li)] = median(s["size"] for s in line["spans"])
                origins[(block["number"], li)] = line["spans"][0]["origin"][1]
    lines = []
    for key, items in words.items():
        items.sort(key=lambda t: t[0].x0)
        rects = [r for r, _ in items]
        h = max(r.height for r in rects)
        size = sizes.get(key) or h * 0.85
        if not 0.5 * h < size < 1.3 * h:  # cỡ chữ vô lý -> ước lượng từ chiều cao
            size = h * 0.85
        base = origins.get(key)
        if base is None or not (min(r.y0 for r in rects) <= base <= max(r.y1 for r in rects) + 1):
            base = max(r.y1 for r in rects) - 0.22 * size
        lines.append((rects, [t for _, t in items], size, base))
    return lines


def background_and_ink(pix_arr, scale, rect):
    """Màu nền (mode của vùng) và màu chữ tương phản. Trả về (bg, fg) RGB 0-1."""
    h, w, _ = pix_arr.shape
    x0, y0 = max(int(rect.x0 * scale), 0), max(int(rect.y0 * scale), 0)
    x1, y1 = min(int(rect.x1 * scale) + 1, w), min(int(rect.y1 * scale) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return (1, 1, 1), (0, 0, 0)
    region = pix_arr[y0:y1, x0:x1].reshape(-1, 3)
    q = (region >> 4).astype(np.int32)
    keys = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    top = Counter(keys.tolist()).most_common(1)[0][0]
    bg = region[keys == top].mean(axis=0) / 255.0
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    if lum > 0.93:
        bg = np.array([1.0, 1.0, 1.0])
    fg = (0, 0, 0) if lum >= 0.5 else (1, 1, 1)
    return tuple(float(c) for c in bg), fg


# --------------------------------------------------------------------------- #
# Xử lý một trang
# --------------------------------------------------------------------------- #
def process_page(page, ctx):
    kind = page_kind(page)
    if kind in ("native", "empty"):
        return kind
    if page.rotation:
        return "skip_rotated"

    # 1) Lấy chữ + vị trí
    if kind == "scan":
        pix = page.get_pixmap(dpi=ctx["ocr_dpi"], alpha=False, annots=False)
        data = pix.pdfocr_tobytes(language=ctx["lang"], tessdata=ctx["tessdata"])
        with pymupdf.open(stream=data) as rec:
            lines = extract_lines(rec[0])
    else:
        lines = extract_lines(page)
    if not lines:
        return "no_text_found"

    # 2) Xóa lớp chữ OCR vô hình cũ (giữ ảnh và nét vẽ)
    if kind == "scan_with_text":
        page.add_redact_annot(page.rect, fill=False)
        page.apply_redactions(images=0, graphics=0, text=0)

    # 3) Mẫu màu nền từ bản render độ phân giải thấp
    scale = 100 / 72
    sample = page.get_pixmap(dpi=100, alpha=False, annots=False)
    arr = np.frombuffer(sample.samples, np.uint8).reshape(sample.height, sample.width, 3)

    font = ctx["font"]
    fname = ctx["fontname"]
    covers, texts = [], []
    for rects, words, size, base in lines:
        # Gạch nối cuối dòng của một số lớp OCR là ký tự lạ "¬" -> đổi thành "-"
        words = [w.replace("\u00ac", "-") for w in words]
        # Từ có ký tự font không hỗ trợ (công thức, chữ Hy Lạp...) -> để nguyên trong ảnh
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
            box = pymupdf.Rect(run[0][0])
            for r, _ in run[1:]:
                box |= r
            text = " ".join(w for _, w in run)
            cover = box + (-1.0, -2.0, 1.0, 3.0)
            bg, fg = background_and_ink(arr, scale, cover)
            covers.append((cover, bg))
            natural = font.text_length(text, fontsize=size)
            sx = box.width / natural if natural > 0 else 1.0
            sx = min(max(sx, 0.6), 1.5)
            texts.append((pymupdf.Point(box.x0, base), text, size, fg, sx))

    # Vẽ hết ô che trước, rồi mới vẽ chữ: ô che của dòng này không đè chữ dòng khác
    for cover, bg in covers:
        page.draw_rect(cover, color=None, fill=bg, overlay=True)
    for origin, text, size, fg, sx in texts:
        page.insert_text(
            origin,
            text,
            fontname=fname,
            fontfile=ctx["fontfile"],
            fontsize=size,
            color=fg,
            morph=(origin, pymupdf.Matrix(sx, 0, 0, 1, 0, 0)),
            overlay=True,
        )
    return "converted"


def make_ctx(args):
    fontfile = args.font if args.font and Path(args.font).exists() else None
    if fontfile:
        font, fname = pymupdf.Font(fontfile=fontfile), "prepfont"
    else:
        font, fname = pymupdf.Font("tiro"), "tiro"  # Times-Roman có sẵn
    return dict(
        lang=args.lang,
        tessdata=get_tessdata(args.lang),
        ocr_dpi=args.dpi,
        font=font,
        fontname=fname,
        fontfile=fontfile,
    )


def worker(job):
    src, part_path, page_numbers, args = job
    ctx = make_ctx(args)
    stats = Counter()
    with pymupdf.open(src) as doc:
        doc.select(page_numbers)
        for i, page in enumerate(doc):
            try:
                stats[process_page(page, ctx)] += 1
            except Exception as exc:  # một trang lỗi không làm hỏng cả file
                stats["error"] += 1
                print(f"  ! trang {page_numbers[i] + 1}: {exc}", flush=True)
        doc.save(part_path, garbage=3, deflate=True)
    return page_numbers[0], part_path, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--pages", help="vd: 1-20 hoặc 1,5,9-12 (mặc định: tất cả)")
    ap.add_argument("--lang", default="eng", help="mã Tesseract, vd eng, eng+deu (mặc định eng)")
    ap.add_argument("--dpi", type=int, default=300, help="DPI khi OCR (mặc định 300)")
    ap.add_argument("--workers", type=int, default=1, help="số tiến trình chạy song song")
    ap.add_argument("--font", help="đường dẫn .ttf (vd C:/Windows/Fonts/times.ttf) để hỗ trợ nhiều ký tự hơn")
    args = ap.parse_args()

    with pymupdf.open(args.input) as d:
        pages = parse_pages(args.pages, len(d))
    n = max(1, args.workers)
    size = -(-len(pages) // n)
    chunks = [pages[i : i + size] for i in range(0, len(pages), size)]
    tmp = tempfile.mkdtemp(prefix="prep_scan_")
    jobs = [(args.input, os.path.join(tmp, f"part{i:03d}.pdf"), c, args) for i, c in enumerate(chunks)]

    t0 = time.time()
    print(f"Xử lý {len(pages)} trang bằng {len(jobs)} tiến trình ...", flush=True)
    if len(jobs) == 1:
        results = [worker(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            results = list(ex.map(worker, jobs))
    results.sort()

    out = pymupdf.open()
    total = Counter()
    for _, part, stats in results:
        total.update(stats)
        with pymupdf.open(part) as p:
            out.insert_pdf(p)
    out.save(args.output, garbage=3, deflate=True)
    print(f"Xong sau {time.time() - t0:.0f}s -> {args.output}")
    names = {
        "converted": "đã chuyển thành chữ thật",
        "native": "chữ gốc, giữ nguyên",
        "empty": "trang trống/ảnh, giữ nguyên",
        "no_text_found": "OCR không thấy chữ",
        "skip_rotated": "trang xoay, bỏ qua",
        "error": "lỗi",
    }
    for k, v in total.items():
        print(f"  {names.get(k, k)}: {v}")


if __name__ == "__main__":
    sys.exit(main())
