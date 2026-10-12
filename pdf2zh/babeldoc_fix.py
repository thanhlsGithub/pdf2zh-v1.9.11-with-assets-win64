"""Vá BabelDOC (chạy lúc runtime, không sửa gói babeldoc): glyph có ToUnicode hỏng
phải được giữ nguyên như "công thức", không được đem đi dịch.

Triệu chứng: bản dịch BabelDOC hiện rác "R%00%00%00’%00 M%00%00..." ở chỗ vốn là cụm
chữ hoa nhỏ (small caps) / ligature của font bị subset.

Nguyên nhân: font có ToUnicode thiếu entry thì pdfminer trả "\\x00" (NUL) cho glyph đó.
BabelDOC chỉ coi một ký tự là công thức khi KHÔNG font nào của nó có glyph cho ký tự
đó (FontMapper.has_char); font Noto lại có glyph "null" cho U+0000 nên NUL bị coi là
chữ thường, gửi thẳng sang bộ dịch và bị mã hoá URL thành "%00" rồi vẽ ra trang.

Cách vá (thay đổi hành vi, không đổi file của babeldoc):
  1. ký tự điều khiển / "%00" / "(cid:N)" luôn là ký tự công thức;
  2. font mà phần lớn glyph trên trang bị lỗi => cả font là "font công thức", để cả
     những ký tự đọc được kẹp giữa (chữ I đầu câu, dấu phẩy...) đi cùng một khối
     công thức và vẽ lại bằng đúng glyph gốc.

Hỗ trợ cả hai bố cục mã nguồn của BabelDOC:
  - 0.1.x / 0.2.x : babeldoc.document_il.midend.styles_and_formulas.StylesAndFormulas
  - >= 0.3.x      : babeldoc.format.pdf.document_il.utils.formular_helper (hàm module)

Dùng: gọi apply() một lần trong tiến trình sẽ chạy BabelDOC (sau khi import babeldoc).
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

_PCT_RE = re.compile(r"(?:%[0-9A-Fa-f]{2})+")


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


def _patch_class_layout(done: list[str]) -> bool:
    """BabelDOC 0.1.x / 0.2.x: logic nằm trong class StylesAndFormulas."""
    try:
        sf = importlib.import_module("babeldoc.document_il.midend.styles_and_formulas")
    except ImportError:
        return False
    cls = getattr(sf, "StylesAndFormulas", None)
    if cls is None or not hasattr(cls, "process_page_formulas"):
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
    done.append("StylesAndFormulas (babeldoc 0.1/0.2 layout)")
    return True


def _patch_function_layout(done: list[str]) -> bool:
    """BabelDOC >= 0.3.x: hàm module trong utils/formular_helper.py."""
    for modname in (
        "babeldoc.format.pdf.document_il.utils.formular_helper",
        "babeldoc.document_il.utils.formular_helper",
    ):
        try:
            fh = importlib.import_module(modname)
            break
        except ImportError:
            fh = None
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
    done.append("formular_helper (babeldoc >= 0.3 layout)")
    return True


def apply() -> list[str]:
    """Vá BabelDOC một lần. Trả về danh sách thứ đã vá (rỗng = không vá được, an toàn bỏ qua)."""
    global _applied
    with _lock:
        if _applied is not None:
            return _applied
        done: list[str] = []
        try:
            if not _patch_class_layout(done):
                _patch_function_layout(done)
        except Exception as e:
            print(f"[pdf2zh-fix] BabelDOC glyph fix failed: {e!r}")
        _applied = done
        if done:
            print(f"[pdf2zh-fix] BabelDOC broken-glyph fix active: {', '.join(done)}")
        else:
            print("[pdf2zh-fix] BabelDOC broken-glyph fix: no known layout found, skipped")
        return done
