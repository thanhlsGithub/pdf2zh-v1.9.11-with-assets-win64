"""Vá BabelDOC lúc chạy (không sửa gói babeldoc). Đặt cạnh gui.py trong thư mục pdf2zh.

Gồm 3 bản vá độc lập, mỗi bản có công tắc môi trường (đặt =0 để tắt):

 1. GLYPH LỖI  (PDF2ZH_BD_GLYPH)   – glyph có ToUnicode hỏng (NUL / "%00" / "(cid:N)") được giữ
    nguyên như công thức, không đem đi dịch. (Bản vá cũ, giữ nguyên.)

 2. PLACEHOLDER CÔNG THỨC (PDF2ZH_BD_PLACEHOLDER) – BabelDOC thay mỗi công thức bằng một chuỗi
    giữ chỗ (với Google/Bing/DeepL... là "<b3></b3>") rồi so khớp CHÍNH XÁC chuỗi đó trong bản dịch.
    Máy dịch hay đổi thành "<b3> </b3>", "< b3 ></ b3 >", viết hoa, xoá một nửa, nhân đôi...
    Khi không khớp, công thức bị rơi khỏi đoạn: mất, hoặc xuất hiện sai chỗ làm chữ đè nhau.
    Bản vá chuẩn hoá lại mọi biến thể về đúng chuỗi gốc; placeholder bị mất hẳn được chèn lại
    theo vị trí tương đối trong câu; bản sao thừa bị xoá.

 3. TÁCH MỤC LỤC / DANH MỤC (PDF2ZH_BD_SPLIT) – BabelDOC chỉ tách mục lục khi dòng trước có >= 20
    dấu chấm LIỀN NHAU (hoặc khi bật split_short_lines, vốn không tách được dòng đầy chữ). Mục
    lục có dấu chấm cách quãng, số trang căn phải, mục lục tra cứu (index) treo lề, danh mục
    tài liệu tham khảo... bị gộp thành một đoạn rồi dàn lại thành một khối. Bản vá thêm bước
    tách theo hình học dòng: (a) dãy dấu chấm dẫn / số trang ở cuối dòng, (b) luật "would-fit"
    (từ đầu dòng sau vẫn vừa chỗ trống cuối dòng trước thì đó là đoạn mới, không phải xuống
    dòng tự động), (c) danh sách treo lề.

Hỗ trợ hai bố cục mã nguồn BabelDOC:
  - 0.1.x .. 0.3.x : babeldoc.document_il.midend.*
  - >= 0.4.x       : babeldoc.format.pdf.document_il.midend.*

Dùng: gọi apply() một lần (sau babeldoc_init()).  PDF2ZH_DIAG=1 để in chẩn đoán.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import threading

_DIAG = bool(os.environ.get("PDF2ZH_DIAG"))
_applied: list[str] | None = None
_lock = threading.Lock()
_tls = threading.local()  # tên font lỗi của trang đang xử lý (theo từng luồng)


def _on(name: str) -> bool:
    return os.environ.get(name, "1") != "0"


_PCT_RE = re.compile(r"(?:%[0-9A-Fa-f]{2})+")


# ==========================================================================
# 1. Glyph lỗi (ToUnicode hỏng)
# ==========================================================================
def is_broken_text(t) -> bool:
    """Text của glyph là rác do ToUnicode hỏng: toàn control char (NUL...), chuỗi giữ
    chỗ "%00"..., hoặc "(cid:N)". Khoảng trắng thật (space, tab, xuống dòng) thì không."""
    if not t or not isinstance(t, str) or t.isspace():
        return False
    if t.startswith("(cid:"):
        return True
    if all(ord(c) < 0x20 or ord(c) == 0x7F for c in t):
        return True
    return bool(re.fullmatch(_PCT_RE, t))


def _broken_font_ids(page, min_broken: int = 3, ratio: float = 0.5) -> set:
    """font_id của các font trên trang mà >= `ratio` glyph (và >= `min_broken` glyph) bị lỗi."""
    stats: dict = {}
    for para in getattr(page, "pdf_paragraph", None) or []:
        for comp in getattr(para, "pdf_paragraph_composition", None) or []:
            line = getattr(comp, "pdf_line", None)
            chars = getattr(line, "pdf_character", None) if line is not None else None
            for ch in chars or []:
                t = getattr(ch, "char_unicode", None)
                if not t or t.isspace():
                    continue
                st = getattr(ch, "pdf_style", None)
                fid = getattr(st, "font_id", None)
                s = stats.setdefault(fid, [0, 0])
                s[1] += 1
                if is_broken_text(t):
                    s[0] += 1
    return {f for f, (nb, tot) in stats.items() if nb >= min_broken and nb >= ratio * tot}


def _replace_everywhere(old, new) -> int:
    """Thay tên `old` đã được import theo kiểu `from x import f` ở mọi module babeldoc."""
    n = 0
    for name, mod in list(sys.modules.items()):
        if not name.startswith("babeldoc") or mod is None:
            continue
        for attr, val in list(vars(mod).items()):
            if val is old:
                setattr(mod, attr, new)
                n += 1
    return n


def _import_first(*names):
    for n in names:
        try:
            return importlib.import_module(n)
        except ImportError:
            continue
    return None


def _patch_class_layout(done: list[str]) -> bool:
    """BabelDOC 0.1.x / 0.2.x: logic nằm trong class StylesAndFormulas."""
    sf = _import_first(
        "babeldoc.document_il.midend.styles_and_formulas",
        "babeldoc.format.pdf.document_il.midend.styles_and_formulas",
    )
    if sf is None:
        return False
    cls = getattr(sf, "StylesAndFormulas", None)
    if cls is None or not hasattr(cls, "process_page_formulas") or not hasattr(cls, "is_formulas_start_char"):
        return False
    if getattr(cls, "_pdf2zh_fix", False):
        return True

    orig_start = cls.is_formulas_start_char
    orig_font = cls.is_formulas_font
    orig_page = cls.process_page_formulas

    def is_formulas_start_char(self, char):
        if is_broken_text(char):
            return True
        return orig_start(self, char)

    def is_formulas_font(self, font_name):
        if orig_font(self, font_name):
            return True
        return font_name in getattr(_tls, "bad_names", ())

    def process_page_formulas(self, page):
        names = set()
        try:
            bad = _broken_font_ids(page)
            if bad:
                names = {f.name for f in (page.pdf_font or []) if f.font_id in bad}
                if _DIAG:
                    print(f"[pdf2zh-fix] BabelDOC page {getattr(page, 'page_number', '?')}: "
                          f"broken-glyph fonts kept as formulas: {sorted(map(str, names))}")
        except Exception as e:  # không bao giờ làm hỏng quá trình dịch
            print(f"[pdf2zh-fix] BabelDOC broken-font scan failed: {e!r}")
        _tls.bad_names = names
        try:
            return orig_page(self, page)
        finally:
            _tls.bad_names = set()

    cls.is_formulas_start_char = is_formulas_start_char
    cls.is_formulas_font = is_formulas_font
    cls.process_page_formulas = process_page_formulas
    cls._pdf2zh_fix = True
    done.append("glyph/StylesAndFormulas")
    return True


def _patch_function_layout(done: list[str]) -> bool:
    """BabelDOC >= 0.3.x: hàm module trong utils/formular_helper.py."""
    fh = _import_first(
        "babeldoc.format.pdf.document_il.utils.formular_helper",
        "babeldoc.document_il.utils.formular_helper",
    )
    if fh is None or not hasattr(fh, "is_formulas_start_char"):
        return False
    if getattr(fh, "_pdf2zh_fix", False):
        return True

    orig_start = fh.is_formulas_start_char

    def is_formulas_start_char(char, *args, **kwargs):
        if is_broken_text(char):
            return True
        return orig_start(char, *args, **kwargs)

    fh.is_formulas_start_char = is_formulas_start_char
    _replace_everywhere(orig_start, is_formulas_start_char)

    orig_collect = getattr(fh, "collect_page_formula_font_ids", None)
    if orig_collect is not None:

        def collect_page_formula_font_ids(page, *args, **kwargs):
            page_ids, xobj_map = orig_collect(page, *args, **kwargs)
            try:
                bad = _broken_font_ids(page)
                if bad:
                    if _DIAG:
                        print(f"[pdf2zh-fix] BabelDOC page {getattr(page, 'page_number', '?')}: "
                              f"broken-glyph font ids kept as formulas: {sorted(map(str, bad))}")
                    page_ids = set(page_ids) | bad
                    xobj_map = {k: set(v) | bad for k, v in xobj_map.items()}
            except Exception as e:
                print(f"[pdf2zh-fix] BabelDOC broken-font scan failed: {e!r}")
            return page_ids, xobj_map

        fh.collect_page_formula_font_ids = collect_page_formula_font_ids
        _replace_everywhere(orig_collect, collect_page_formula_font_ids)

    fh._pdf2zh_fix = True
    done.append("glyph/formular_helper")
    return True


# ==========================================================================
# 2. Khôi phục placeholder công thức bị máy dịch làm hỏng
# ==========================================================================
def _tolerant_pattern(ph: str) -> str:
    """"<b3></b3>" -> regex chấp nhận khoảng trắng chen giữa mọi ký hiệu, không phân biệt
    hoa/thường, và số 3 không dính vào "30"."""
    toks = re.findall(r"\d+|[A-Za-z]+|[^\sA-Za-z\d]", ph)
    parts = []
    for t in toks:
        if t.isdigit():
            parts.append(r"(?<!\d)" + re.escape(t) + r"(?!\d)")
        else:
            parts.append(re.escape(t))
    return r"\s*".join(parts)


def restore_formula_placeholders(src: str, out: str, phs: list[str]):
    """src: chuỗi gốc gửi đi dịch (có placeholder). out: bản dịch. phs: các placeholder công
    thức CHÍNH XÁC. Trả về (out mới, số cái đã sửa, số cái chèn lại, số bản sao đã xoá)."""
    fixed = inserted = removed = 0
    for ph in phs:
        if not ph:
            continue
        ms = list(re.finditer(_tolerant_pattern(ph), out, re.IGNORECASE))
        if ms:
            pieces, last = [], 0
            for k, m in enumerate(ms):
                pieces.append(out[last:m.start()])
                if k == 0:
                    pieces.append(ph)
                    if m.group(0) != ph:
                        fixed += 1
                else:
                    removed += 1
                last = m.end()
            pieces.append(out[last:])
            out = "".join(pieces)
            continue
        # mất hẳn: chèn lại theo vị trí tương đối trong câu gốc
        p = src.find(ph)
        rel = (p / max(len(src), 1)) if p >= 0 else 1.0
        pos = int(rel * len(out))
        nxt = out.find(" ", pos)
        pos = len(out) if nxt < 0 else nxt
        left = "" if pos == 0 or out[pos - 1].isspace() else " "
        right = "" if pos >= len(out) or out[pos].isspace() else " "
        out = out[:pos] + left + ph + right + out[pos:]
        inserted += 1
    return out, fixed, inserted, removed


def _patch_placeholders(done: list[str]) -> bool:
    mod = _import_first(
        "babeldoc.document_il.midend.il_translator",
        "babeldoc.format.pdf.document_il.midend.il_translator",
    )
    cls = getattr(mod, "ILTranslator", None) if mod else None
    if cls is None or not hasattr(cls, "parse_translate_output"):
        return False
    if getattr(cls, "_pdf2zh_ph_fix", False):
        return True
    orig = cls.parse_translate_output

    def parse_translate_output(self, input_text, output):
        try:
            phs = [p.placeholder for p in (input_text.placeholders or []) if hasattr(p, "placeholder")]
            if phs and isinstance(output, str):
                new, nf, ni, nr = restore_formula_placeholders(getattr(input_text, "unicode", "") or "", output, phs)
                if _DIAG and (nf or ni or nr):
                    print(f"[pdf2zh-fix] BabelDOC placeholders: repaired={nf} re-inserted={ni} duplicates-removed={nr} "
                          f"| {output[:80]!r}")
                output = new
        except Exception as e:  # không bao giờ làm hỏng quá trình dịch
            print(f"[pdf2zh-fix] BabelDOC placeholder repair failed: {e!r}")
        return orig(self, input_text, output)

    cls.parse_translate_output = parse_translate_output
    cls._pdf2zh_ph_fix = True
    done.append("placeholder/ILTranslator")
    return True


# ==========================================================================
# 3. Tách mục lục / danh mục theo hình học dòng
# ==========================================================================
_LEADER_RE = re.compile(r"(?:[.\u00b7\u2022\u2026_][\s\u00a0]*){6,}")
_HYPHENS = ("-", "\u2010", "\u2011", "\u00ad")
_PAGENO_RE = re.compile(r"[\divxlcdmIVXLCDM][\divxlcdmIVXLCDM\-\u2013,]*")


def _line_infos(comps):
    """Thông tin hình học từng dòng; None nếu thành phần không phải dòng chữ."""
    infos = []
    for comp in comps:
        line = getattr(comp, "pdf_line", None)
        chars = list(getattr(line, "pdf_character", None) or []) if line is not None else []
        vis = [c for c in chars if c.char_unicode and not c.char_unicode.isspace()]
        if not vis:
            infos.append(None)
            continue
        sizes = sorted(c.pdf_style.font_size for c in vis if c.pdf_style and c.pdf_style.font_size)
        size = sizes[len(sizes) // 2] if sizes else max(line.box.y2 - line.box.y, 1.0)
        first = vis[0]
        tok_x2 = first.box.x2
        for c in vis[1:]:                                   # từ đầu dòng: dừng khi có khoảng hở
            if c.box.x - tok_x2 > 0.18 * size:
                break
            tok_x2 = max(tok_x2, c.box.x2)
        # "đuôi số trang": từ cuối ngắn nằm xa phần chữ phía trước
        tail = False
        k = len(vis) - 1
        end_x2 = vis[k].box.x2
        start = k
        while start > 0 and vis[start - 1].box.x2 >= vis[start].box.x - 0.18 * size:
            start -= 1
        if start > 0:
            word = "".join(c.char_unicode for c in vis[start:])
            gap = vis[start].box.x - vis[start - 1].box.x2
            tail = len(word) <= 8 and gap > 2.5 * size and bool(_PAGENO_RE.fullmatch(word))
        infos.append({
            "x0": min(c.box.x for c in vis), "x1": max(c.box.x2 for c in vis), "size": size,
            "tok_w": tok_x2 - first.box.x, "first": first.char_unicode, "last": vis[-1].char_unicode,
            "text": "".join(c.char_unicode or "" for c in chars), "tail": tail,
        })
    return infos


def smart_cuts(comps, slack: float = 0.4) -> list[int]:
    """Chỉ số dòng j (j>=1) mà tại đó bắt đầu một đoạn/mục MỚI."""
    infos = _line_infos(comps)
    n = len(infos)
    if n < 2 or any(i is None for i in infos):
        return []
    right = sorted(i["x1"] for i in infos)[min(n - 1, int(0.95 * n))]
    left = sorted(i["x0"] for i in infos)[int(0.15 * (n - 1))]
    fits = [right - a["x1"] >= b["tok_w"] + slack * b["size"] for a, b in zip(infos, infos[1:])]
    ind = [i for i in range(1, n) if infos[i]["x0"] - left >= 0.8 * infos[i]["size"]]
    full_before = [i for i in ind if not fits[i - 1]]
    hang = len(full_before) >= 2 and len(full_before) >= 0.5 * len(ind)
    cuts = []
    for j in range(1, n):
        a, b = infos[j - 1], infos[j]
        sz = b["size"]
        if _LEADER_RE.search(a["text"]) or a["tail"]:               # dấu chấm dẫn / số trang cuối dòng
            cuts.append(j)
            continue
        if hang and (b["x0"] - left < 0.5 * sz or b["x0"] < a["x0"] - 0.5 * sz):
            cuts.append(j)                                          # treo lề: dòng sát lề / ra ngoài hơn dòng trước
            continue
        if b["first"].islower() or a["last"] in _HYPHENS:           # câu còn tiếp / ngắt từ
            continue
        if abs((a["x0"] + a["x1"]) / 2 - (b["x0"] + b["x1"]) / 2) < 0.6 * sz and abs(a["x0"] - b["x0"]) > 0.6 * sz:
            continue                                                # khối căn giữa
        if fits[j - 1]:
            cuts.append(j)
    return cuts


def _patch_paragraphs(done: list[str]) -> bool:
    mod = _import_first(
        "babeldoc.document_il.midend.paragraph_finder",
        "babeldoc.format.pdf.document_il.midend.paragraph_finder",
    )
    cls = getattr(mod, "ParagraphFinder", None) if mod else None
    if cls is None or not hasattr(cls, "process_independent_paragraphs"):
        return False
    if getattr(cls, "_pdf2zh_split_fix", False):
        return True
    orig = cls.process_independent_paragraphs
    PdfParagraph, Box = getattr(mod, "PdfParagraph", None), getattr(mod, "Box", None)
    new_id = getattr(mod, "generate_base58_id", lambda *a, **k: "")
    if PdfParagraph is None or Box is None:
        return False

    def process_independent_paragraphs(self, paragraphs, median_width, *args, **kwargs):
        try:
            i, n_split = 0, 0
            while i < len(paragraphs):
                p = paragraphs[i]
                comps = list(p.pdf_paragraph_composition or [])
                cuts = smart_cuts(comps) if len(comps) > 1 else []
                if cuts:
                    bounds = [0] + cuts + [len(comps)]
                    parts = [comps[a:b] for a, b in zip(bounds, bounds[1:])]
                    p.pdf_paragraph_composition = parts[0]
                    self.update_paragraph_data(p)
                    for k, part in enumerate(parts[1:], 1):
                        q = PdfParagraph(
                            box=Box(0, 0, 0, 0), pdf_paragraph_composition=part, unicode="",
                            debug_id=new_id(), layout_label=p.layout_label,
                        )
                        self.update_paragraph_data(q)
                        paragraphs.insert(i + k, q)
                    n_split += len(parts) - 1
                    i += len(parts)
                else:
                    i += 1
            if _DIAG and n_split:
                print(f"[pdf2zh-fix] BabelDOC split {n_split} extra entries (TOC / index / list)")
        except Exception as e:  # không bao giờ làm hỏng quá trình dịch
            print(f"[pdf2zh-fix] BabelDOC paragraph split failed: {e!r}")
        return orig(self, paragraphs, median_width, *args, **kwargs)

    cls.process_independent_paragraphs = process_independent_paragraphs
    cls._pdf2zh_split_fix = True
    done.append("split/ParagraphFinder")
    return True


# ==========================================================================
def apply() -> list[str]:
    """Vá BabelDOC một lần. Trả về danh sách thứ đã vá (rỗng = không vá được, an toàn bỏ qua)."""
    global _applied
    with _lock:
        if _applied is not None:
            return _applied
        done: list[str] = []
        for flag, fn in (
            ("PDF2ZH_BD_PLACEHOLDER", _patch_placeholders),
            ("PDF2ZH_BD_SPLIT", _patch_paragraphs),
        ):
            try:
                if _on(flag):
                    fn(done)
            except Exception as e:
                print(f"[pdf2zh-fix] BabelDOC patch {fn.__name__} failed: {e!r}")
        try:
            if _on("PDF2ZH_BD_GLYPH") and not _patch_class_layout(done):
                _patch_function_layout(done)
        except Exception as e:
            print(f"[pdf2zh-fix] BabelDOC glyph fix failed: {e!r}")
        _applied = done
        if done:
            print(f"[pdf2zh-fix] BabelDOC fixes active: {', '.join(done)}")
        else:
            print("[pdf2zh-fix] BabelDOC fixes: no known layout found, skipped")
        return done