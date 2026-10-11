import concurrent.futures
import difflib
import logging
import os
import re
import unicodedata
from collections import Counter
from enum import Enum
from string import Template
from typing import Dict

import numpy as np
from pdfminer.converter import PDFConverter
from pdfminer.layout import LTChar, LTFigure, LTLine, LTPage
from pdfminer.pdffont import PDFCIDFont, PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager
from pdfminer.utils import apply_matrix_pt, mult_matrix
from pymupdf import Document, Font, Matrix, csRGB
from tenacity import retry, wait_fixed

from pdf2zh.translator import (
    AnythingLLMTranslator,
    ArgosTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DeepseekTranslator,
    DifyTranslator,
    GeminiTranslator,
    GoogleTranslator,
    GoogleNewTranslator,
    GoogleOldTranslator,
    GoogleHtmlTranslator,
    EdgeTranslator,
    GrokTranslator,
    GroqTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAIlikedTranslator,
    OpenAITranslator,
    QwenMtTranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
)

log = logging.getLogger(__name__)
_obj_id = id  # receive_layout dùng `id` làm biến vòng lặp nên giữ bí danh của hàm id()

# ==========================================================================
# Hàm phụ: tách đoạn (would-fit / treo lề) và giữ màu theo từng đoạn chữ
# (trước đây nằm ở fixlib.py, nay gộp thẳng vào converter.py)
# ==========================================================================
_HYPHENS = ("-", "\u2010", "\u2011", "\u00ad")


# --------------------------------------------------------------------------
# 1. Tách đoạn
# --------------------------------------------------------------------------
def _collect_lines(chars, cls_of, skip):
    """Gom ký tự thành các dòng, theo từng lớp layout (thứ tự content stream)."""
    lines, last = {}, {}
    for ch in chars:
        if not isinstance(ch, LTChar):
            continue
        t = ch.get_text()
        if not t or t.isspace():
            continue
        c = int(cls_of(ch))
        if c < 2 or c in skip:      # 0 = công thức/hình, 1 = ngoài mọi khung
            continue
        p = last.get(c)
        last[c] = ch
        new_line = p is not None and (
            (ch.x1 < p.x0 and p.y0 - ch.y0 > 0.5 * ch.size) or p.y0 - ch.y0 > 0.9 * ch.size
        )
        if p is not None and not new_line:
            ln = lines[c][-1]
            ln["x1"] = max(ln["x1"], ch.x1)
            ln["last"] = ch
            if not ln["tok_done"]:
                if ch.x0 - ln["tok_x1"] > 0.18 * ch.size:
                    ln["tok_done"] = True
                else:
                    ln["tok_x1"] = max(ln["tok_x1"], ch.x1)
            continue
        lines.setdefault(c, []).append(
            {"first": ch, "last": ch, "x0": ch.x0, "x1": ch.x1, "tok_x1": ch.x1,
             "tok_done": False, "size": ch.size}
        )
    return lines


def reflow_plan(chars, cls_of, skip=(), slack: float = 0.4):
    """chars  : các LTChar theo thứ tự content stream.
    cls_of    : hàm ch -> lớp layout (cùng cách tính như converter).
    skip      : tập lớp bỏ qua.
    Trả về (breaks, hang_cls):
      breaks   : tập id(ch) của ký tự ĐẦU DÒNG mở đầu một đoạn/mục mới.
      hang_cls : tập lớp layout có kiểu TREO LỀ (dòng nối tiếp thụt vào so với dòng đầu mục).

    Luật would-fit: giữa dòng A và dòng B (cùng khung layout) có ranh giới đoạn nếu từ
    đầu của B VẪN vừa chỗ trống còn lại cuối dòng A (nếu vừa mà không dồn lên thì đó
    không phải xuống dòng tự động). Bỏ qua khi A kết thúc bằng dấu nối, B bắt đầu bằng
    chữ thường (câu còn tiếp), hoặc A/B là khối căn giữa.

    Luật treo lề: khung mà phần lớn dòng thụt vào đều đứng ngay sau một dòng ĐẦY
    (= dòng nối tiếp) thì là danh sách treo lề; trong khung đó dòng sát lề trái, hoặc
    dòng thụt ít hơn dòng trước, luôn mở đầu một mục mới."""
    lines = _collect_lines(chars, cls_of, skip)
    out, hang_cls = set(), set()
    for c, ls in lines.items():
        n = len(ls)
        if n < 2:
            continue
        xs = sorted(l["x1"] for l in ls)
        right = xs[min(n - 1, int(0.95 * n))]               # mép phải của khung (bỏ qua dòng lòi ra)
        left = sorted(l["x0"] for l in ls)[int(0.15 * (n - 1))]   # mép trái của khung
        fits = []                                           # fits[i-1]: từ đầu dòng i vừa chỗ trống cuối dòng i-1
        for a, b in zip(ls, ls[1:]):
            tok_w = b["tok_x1"] - b["x0"]
            fits.append(right - a["x1"] >= tok_w + slack * b["size"])
        ind = [i for i in range(1, n) if ls[i]["x0"] - left >= 0.8 * ls[i]["size"]]
        full_before = [i for i in ind if not fits[i - 1]]
        hang = len(full_before) >= 2 and len(full_before) >= 0.5 * len(ind)
        if hang:
            hang_cls.add(c)
        for i in range(1, n):
            a, b = ls[i - 1], ls[i]
            sz = b["size"]
            if hang and (b["x0"] - left < 0.5 * sz or b["x0"] < a["x0"] - 0.5 * sz):
                out.add(id(b["first"]))                     # treo lề: dòng sát lề / dòng ra ngoài hơn dòng trước
                continue
            if b["first"].get_text().islower():             # câu còn tiếp
                continue
            if a["last"].get_text() in _HYPHENS:            # ngắt từ bằng dấu nối
                continue
            if abs((a["x0"] + a["x1"]) / 2 - (b["x0"] + b["x1"]) / 2) < 0.6 * sz \
                    and abs(a["x0"] - b["x0"]) > 0.6 * sz:  # khối căn giữa
                continue
            if fits[i - 1]:
                out.add(id(b["first"]))
    return out, hang_cls


def reflow_breaks(chars, cls_of, skip=(), slack: float = 0.4) -> set:
    """Chỉ trả về tập ký tự đầu mục mới (xem reflow_plan)."""
    return reflow_plan(chars, cls_of, skip, slack)[0]


# --------------------------------------------------------------------------
# 1b. Gộp ký tự "đệm" giữa hai glyph lỗi vào chung một khối công thức
# --------------------------------------------------------------------------
_FILLER_CHARS = set(" '\u2019\u2018\"\u201c\u201d-\u2013\u2014")


def broken_glyph_merge_ids(chars, cls_of, vflag) -> set:
    """Trả về id() các ký tự "đệm" (dấu cách, nháy, gạch nối...) nằm giữa hai
    ký tự glyph lỗi (vflag() trả True — ví dụ ToUnicode CMap hỏng trả control
    char, xem vflag()), trong cùng một khung layout.

    Lý do cần hàm này: nếu không gộp, mỗi lần gặp một ký tự "đệm" giữa hai
    glyph lỗi, converter sẽ đóng khối công thức đang dở và mở một placeholder
    {vN} MỚI ngay sau đó. Một cụm chữ hoa nhỏ (small caps) bị vỡ glyph dài có
    dấu cách/dấu nháy bên trong (ví dụ một câu trích dẫn) vì vậy bị cắt thành
    HÀNG CHỤC placeholder {vN} rời rạc, tất cả đều bị nhét thẳng vào câu gửi
    đi dịch máy. Càng nhiều placeholder rời rạc, xác suất bộ dịch làm hỏng cú
    pháp của một trong số đó càng cao — và khi việc đó xảy ra, placeholder đó
    không khớp lại được lúc dựng trang, nên hiện ra thành chữ thô "{vN}" trên
    trang thay vì được thay bằng glyph gốc. Gộp các ký tự đệm vào chung một
    khối giảm mạnh số lượng placeholder, giảm rủi ro này."""
    n = len(chars)
    out = set()
    i = 0
    while i < n:
        if chars[i].get_text() in _FILLER_CHARS:
            j = i
            while j < n and chars[j].get_text() in _FILLER_CHARS:
                j += 1
            if i > 0 and j < n and cls_of(chars[i - 1]) == cls_of(chars[j]):
                if vflag(chars[i - 1].fontname, chars[i - 1].get_text()) and vflag(chars[j].fontname, chars[j].get_text()):
                    out.update(id(c) for c in chars[i:j])
            i = j
        else:
            i += 1
    return out


# --------------------------------------------------------------------------
# 2. Màu theo từng đoạn chữ
# --------------------------------------------------------------------------
def op_rgb(op):
    try:
        t = op.split()
        v = [float(x) for x in t[:-1]]
        kind = t[-1]
    except Exception:
        return None
    if kind == "g" and len(v) == 1:
        return (v[0],) * 3
    if kind == "rg" and len(v) == 3:
        return tuple(v)
    if kind == "k" and len(v) == 4:
        c, m, y, k = v
        return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))
    return None


def colour_key(op):
    """Khoá màu (bước 0.05): 0 g, 0 0 0 rg, 0.02 g coi là cùng một màu; hai màu nhìn khác nhau thì không bị gộp."""
    if op is None:
        return None
    rgb = op_rgb(op)
    if rgb is None:
        return op
    return tuple(int(round(min(max(x, 0.0), 1.0) * 20)) for x in rgb)


def fill_unknown(recs, sample):
    """recs  : [(idx, độ_dài, op|None, box|None)].
    sample   : hàm (list box) -> op | None  (lấy mẫu pixel từ trang gốc).
    Ký tự có op=None (màu pattern/separation/...) được lấy mẫu THEO TỪNG TỪ
    (các ký tự liền nhau), nên chữ xanh và chữ đen cùng không đọc được từ lệnh
    vẫn tách được thành hai màu. Trả về recs đã điền op."""
    out, i, n = [], 0, len(recs)
    while i < n:
        r = recs[i]
        if r[2] is not None:
            out.append(r)
            i += 1
            continue
        j = i + 1
        while j < n and recs[j][2] is None and recs[j][0] == recs[j - 1][0] + recs[j - 1][1]:
            j += 1
        op = sample([x[3] for x in recs[i:j] if x[3]]) or "0 g"
        out.extend((x[0], x[1], op, None, x[4] if len(x) > 4 else 0) for x in recs[i:j])
        i = j
    return out


def pick_op(recs):
    """Lệnh màu đại diện của một run: màu xuất hiện nhiều nhất."""
    c = Counter(r[2] for r in recs if r[2])
    return c.most_common(1)[0][0] if c else "0 g"


def make_segments(text, recs, resolve):
    """text : chuỗi nguồn của đoạn (sstk[i]).
    recs    : [(idx, độ_dài, op|None, box|None, style)] ghi tại lúc đọc từng ký tự chữ.
              style: bit 1 = đậm, bit 2 = nghiêng.
    resolve : hàm (recs của một run) -> chuỗi lệnh màu.
    Trả về [(start, end, op, key)] – các run màu liên tiếp, đã gộp dấu câu kẹp giữa
    hai run cùng màu (ví dụ "6" xanh , "," đen , "7" xanh -> một run xanh)."""
    n = len(text)
    recs = [r for r in recs if r[0] < n]
    if not recs:
        return []
    canon = []                      # màu chỉ lệch ≤ 1 bước (0.05) coi là cùng một màu (sai số lấy mẫu pixel)

    def canon_of(k):
        if not isinstance(k, tuple):
            return k
        for c in canon:
            if len(c) == len(k) and all(abs(a - b) <= 1 for a, b in zip(c, k)):
                return c
        canon.append(k)
        return k

    groups = []
    for r in recs:
        k = (canon_of(colour_key(r[2])), r[4] if len(r) > 4 else 0)   # (màu, style: 1 = đậm, 2 = nghiêng)
        if groups and groups[-1]["k"] == k:
            g = groups[-1]
            g["end"] = min(r[0] + r[1], n)
            g["recs"].append(r)
        else:
            groups.append({"k": k, "s": r[0], "end": min(r[0] + r[1], n), "recs": [r]})
    merged = True
    while merged and len(groups) >= 3:
        merged = False
        for i in range(1, len(groups) - 1):
            g = groups[i]
            if groups[i - 1]["k"] == groups[i + 1]["k"] and not re.search(r"\w", text[g["s"]:g["end"]]):
                a, b = groups[i - 1], groups[i + 1]
                a["end"] = b["end"]
                a["recs"] = a["recs"] + g["recs"] + b["recs"]
                del groups[i:i + 2]
                merged = True
                break
    return [(g["s"], g["end"], (resolve(g["recs"]), g["k"][1]), g["k"]) for g in groups]


def plan_runs(segs, text=None):
    """-> (op màu nền, [(start, end, op) các run khác màu nền]).
    Màu nền = màu của phần CHỮ CÁI chiếm đa số (nếu có `text`), để mục lục kiểu
    "tên mục (đen) + dãy số trang (xanh)" lấy màu đen làm nền dù số trang dài hơn;
    nếu một run không định vị được trong bản dịch thì phần chữ vẫn giữ màu chữ."""
    if not segs:
        return None, []
    tot, first = Counter(), {}
    for s, e, op, k in segs:
        w = sum(1 for ch in text[s:e] if ch.isalpha()) if text is not None else e - s
        tot[k] += w
        first.setdefault(k, op)
    if text is not None and not any(tot.values()):
        for s, e, op, k in segs:
            tot[k] += e - s
    major = tot.most_common(1)[0][0]
    return first[major], [(s, e, op) for s, e, op, k in segs if k != major]


def needs_translation(t: str) -> bool:
    """Run có chữ cái -> cần dịch riêng để định vị trong bản dịch của cả đoạn."""
    return (
        bool(re.search(r"[^\W\d_]", t))
        and len(t) <= 80
        and not re.fullmatch(r"(\{\s*v[\d\s]+\}\s*)+", t)
    )


def _norm(s: str) -> str:
    out = "".join(c.lower() if len(c.lower()) == 1 else c for c in s)
    return re.sub(r"[.,]", ".", out)       # 0.5 và 0,5 coi như nhau (dịch sang tiếng Việt đổi dấu thập phân)


def find_exact(hay: str, needle: str, start: int = 0):
    """Khớp chính xác (không phân biệt hoa/thường, . ,) -> (start, end) | None."""
    n = needle.strip()
    if not n:
        return None
    H, N = _norm(hay), _norm(n)
    pre = r"(?<!\w)" if N[0].isalnum() else ""
    suf = r"(?!\w)" if N[-1].isalnum() else ""
    pat = re.compile(pre + re.escape(N) + suf)
    m = pat.search(H, start) or pat.search(H)
    return (m.start(), m.end()) if m else None


def locate(hay: str, needle: str, start: int = 0):
    """Tìm needle trong hay: khớp chính xác, rồi khớp mờ."""
    hit = find_exact(hay, needle, start)
    if hit:
        return hit
    n = needle.strip()
    if not n:
        return None
    H, N = _norm(hay), _norm(n)
    b = difflib.SequenceMatcher(None, H, N, autojunk=False).find_longest_match(0, len(H), 0, len(N))
    if b.size >= max(3, int(0.8 * len(N))):
        s = max(b.a - b.b, 0)
        return s, min(s + len(N), len(H))
    return None


def locate_spans(new, src, runs, tr):
    """Định vị các run khác màu của bản gốc trong bản dịch `new`.
    tr: {văn bản run: bản dịch riêng của run}. Trả [(start, end, op)] không chồng lấn."""
    out, pos = [], 0
    for s, e, op in runs:
        t = src[s:e].strip()
        hit = locate(new, tr.get(t, t), pos)
        if hit and hit[1] > hit[0] and hit[0] >= pos:
            out.append((hit[0], hit[1], op))
            pos = hit[1]
    return out


def colour_map(new, base, spans):
    """Màu cho từng ký tự của bản dịch."""
    cm = [base] * len(new)
    for s, e, op in spans:
        cm[s:e] = [op] * (e - s)
    return cm



class PDFConverterEx(PDFConverter):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, None, "utf-8", 1, None)

    def begin_page(self, page, ctm) -> None:
        # 重载替换 cropbox
        (x0, y0, x1, y1) = page.cropbox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(page.pageno, mediabox)
        self._pdfdoc = getattr(page, "doc", None)  # used to render the original page (colour sampling)

    def end_page(self, page):
        # 重载返回指令流
        return self.receive_layout(self.cur_item)

    def begin_figure(self, name, bbox, matrix) -> None:
        # 重载设置 pageid
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, _: str) -> None:
        # 重载返回指令流
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)
        return self.receive_layout(fig)

    def render_char(
        self,
        matrix,
        font,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs,
        graphicstate: PDFGraphicState,
    ) -> float:
        # 重载设置 cid 和 font
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        item.cid = cid  # hack 插入原字符编码
        item.font = font  # hack 插入原字符字体
        return item.adv


_COLOR_SPACES = (
    "DeviceGray", "CalGray", "DeviceRGB", "CalRGB", "DeviceCMYK", "ICCBased",
)


def _fill_color_op(ch) -> str | None:
    """PDF operator that sets the fill color to the color of the original
    character `ch` (e.g. "0 0.2 0.8 rg"), or None if it cannot be determined
    (pattern, separation, indexed... colors)."""
    try:
        name = getattr(ch.ncs, "name", None)
        color = ch.graphicstate.ncolor
        if name not in _COLOR_SPACES:
            return None
        comps = [color] if isinstance(color, (int, float)) else list(color)
        comps = [min(max(float(v), 0.0), 1.0) for v in comps]
    except Exception:
        return None
    if len(comps) == 1:
        return f"{comps[0]:.3f} g"
    if len(comps) == 3:
        return "{:.3f} {:.3f} {:.3f} rg".format(*comps)
    if len(comps) == 4:
        return "{:.3f} {:.3f} {:.3f} {:.3f} k".format(*comps)
    return None


_BOLD_RE = re.compile(r"(bold|black|heavy|demi|[-,_ ]bd(?:[-,_ ]|$)|[-,_ ]b(?:[-,_ ]|$))", re.I)
_ITAL_RE = re.compile(r"(ital|obli|slant|kursiv|[-,_ ]it(?:[-,_ ]|$)|[-,_ ]i(?:[-,_ ]|$))", re.I)
_style_cache: dict = {}


def _style_of(ch) -> int:
    """Kiểu chữ của ký tự gốc: bit 1 = đậm, bit 2 = nghiêng (theo tên font và font descriptor)."""
    name = str(getattr(ch, "fontname", "") or "")
    if isinstance(getattr(ch, "fontname", None), bytes):
        name = ch.fontname.decode("latin-1", "ignore")
    key = (name, id(getattr(ch, "font", None)))
    if key in _style_cache:
        return _style_cache[key]
    base = name.split("+")[-1]
    bold = bool(_BOLD_RE.search(base))
    ital = bool(_ITAL_RE.search(base) or re.search(r"[a-z]It$", base))
    try:
        d = getattr(getattr(ch, "font", None), "descriptor", None) or {}
        flags = int(d.get("Flags", 0) or 0)
        if flags & 0x40:                                    # Italic
            ital = True
        if flags & 0x40000:                                 # ForceBold
            bold = True
        if float(d.get("ItalicAngle", 0) or 0) != 0:
            ital = True
        if float(d.get("FontWeight", 0) or 0) >= 600:
            bold = True
    except Exception:
        pass
    _style_cache[key] = (1 if bold else 0) | (2 if ital else 0)
    return _style_cache[key]


# Set the environment variable PDF2ZH_DIAG=1 to print one diagnostic line per page.
_DIAG = bool(os.environ.get("PDF2ZH_DIAG"))
_FIX_VERSION = "converter fix v9 single-file + bold/italic (per-run colours + would-fit/hanging-indent entry split + hanging indent + overlap guard + broken-glyph filler merge)"
_banner_shown = False


class Paragraph:
    def __init__(self, y, x, x0, x1, y0, y1, size, brk):
        self.y: float = y  # 初始纵坐标
        self.x: float = x  # 初始横坐标
        self.x0: float = x0  # 左边界
        self.x1: float = x1  # 右边界
        self.y0: float = y0  # 上边界
        self.y1: float = y1  # 下边界
        self.size: float = size  # 字体大小
        self.brk: bool = brk  # 换行标记
        self.cx: float = None  # x của dòng nối tiếp đầu tiên (dùng cho danh sách treo lề)


def _ov(a0: float, a1: float, b0: float, b1: float) -> float:
    """Độ dài phần giao của hai đoạn [a0,a1] và [b0,b1] (âm = không giao)."""
    return min(a1, b1) - max(a0, b0)


def spaced_list_boxes(ltpage, layout) -> set:
    """Tìm các khung layout có dạng DANH SÁCH / MỤC LỤC: nhiều dòng ngắn, cách
    nhau bằng khoảng trắng lớn. Văn xuôi (dù giãn dòng đôi) có các dòng gần
    chạm lề phải nên không bị nhận nhầm. Trong các khung này mỗi mục sẽ được
    tách thành một đoạn riêng (xem `gap_end` trong receive_layout)."""
    h, w = layout.shape
    lines: dict = {}
    last: dict = {}
    for ch in ltpage:
        if not isinstance(ch, LTChar) or ch.get_text().isspace():
            continue
        c = int(layout[np.clip(int(ch.y0), 0, h - 1), np.clip(int(ch.x0), 0, w - 1)])
        if c < 2:                       # 0 = công thức/hình, 1 = ngoài mọi khung
            continue
        p = last.get(c)
        last[c] = ch
        if p is not None and not (ch.x1 < p.x0 and p.y0 - ch.y0 > 0.5 * ch.size):
            ln = lines[c][-1]           # cùng dòng
            ln[0] = min(ln[0], ch.x0)
            ln[1] = max(ln[1], ch.x1)
            continue
        lines.setdefault(c, []).append([ch.x0, ch.x1, ch.y0, ch.size])  # dòng mới
    out = set()
    for c, ls in lines.items():
        n = len(ls)
        if n < 3:
            continue
        right = max(l[1] for l in ls)
        spaced = sum(1 for a, b in zip(ls, ls[1:]) if a[2] - b[2] > 1.7 * a[3])
        short = sum(1 for l in ls[:-1] if l[1] < right - 4 * l[3])
        if spaced >= 0.3 * (n - 1) and short >= 0.4 * (n - 1):
            out.add(c)
    return out


def free_right(pstk, i: int, page_w: float) -> float:
    """Toạ độ x xa nhất mà đoạn 1 dòng `i` được phép vươn tới bên phải mà không
    đè lên đoạn khác (hoặc ra ngoài lề trang). Không bao giờ nhỏ hơn mép phải
    của nguyên bản."""
    p = pstk[i]
    gap = 0.3 * p.size
    lim = page_w - max(0.03 * page_w, p.size)
    b0, b1 = p.y0, p.y1
    for j, q in enumerate(pstk):
        if j == i:
            continue
        # dòng đầu của q (kể cả khi q là đoạn nhiều dòng có chữ cái rơi/thụt dòng đầu)
        if q.x >= p.x1 - gap and _ov(b0, b1, q.y, q.y + q.size) > 0.3 * min(b1 - b0, q.size):
            lim = min(lim, q.x - gap)
        # toàn bộ khung của q nằm bên phải
        if q.x0 >= p.x1 - gap and _ov(b0, b1, q.y0, q.y1) > 0.3 * min(b1 - b0, q.y1 - q.y0):
            lim = min(lim, q.x0 - gap)
    return max(lim, p.x1)


def left_float_end(pstk, i: int, y_line: float, size: float, x0: float) -> float:
    """Điểm bắt đầu của một dòng (thụt vào) khi bên trái cột có vật nổi hẹp
    (chữ cái đầu đoạn to - drop cap, số thứ tự lớn...) chiếm dòng đó."""
    p = pstk[i]
    half = x0 + 0.5 * (p.x1 - p.x0)
    start = x0
    b0, b1 = y_line, y_line + size
    for j, q in enumerate(pstk):
        if j == i:
            continue
        if (
            q.x1 > x0 + 0.5 * size      # thật sự lấn vào cột
            and q.x0 < half
            and q.x1 <= half            # hẹp hơn nửa cột
            and _ov(b0, b1, q.y0, q.y1) > 0.5 * size
        ):
            start = max(start, q.x1 + 0.4 * size)
    return start


# fmt: off
class TranslateConverter(PDFConverterEx):
    def __init__(
        self,
        rsrcmgr,
        vfont: str = None,
        vchar: str = None,
        thread: int = 0,
        layout={},
        lang_in: str = "",
        lang_out: str = "",
        service: str = "",
        noto_name: str = "",
        noto: Font = None,
        envs: Dict = None,
        prompt: Template = None,
        ignore_cache: bool = False,
    ) -> None:
        super().__init__(rsrcmgr)
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.layout = layout
        self.noto_name = noto_name
        self.noto = noto
        self.translator: BaseTranslator = None
        self._pdfdoc = None
        self._mdoc = None        # PyMuPDF copy of the original PDF (only opened when needed)
        self._mdoc_fp = None
        self._ras = None         # (pageid, rgb array, scale, page height) of the page being laid out
        global _banner_shown
        if not _banner_shown:
            _banner_shown = True
            print(f"[pdf2zh-fix] {_FIX_VERSION} is active")
        # e.g. "ollama:gemma2:9b" -> ["ollama", "gemma2:9b"]
        param = service.split(":", 1)
        service_name = param[0]
        service_model = param[1] if len(param) > 1 else None
        if not envs:
            envs = {}
        for translator in [GoogleTranslator, GoogleNewTranslator, GoogleOldTranslator, GoogleHtmlTranslator, BingTranslator, EdgeTranslator, DeepLTranslator, DeepLXTranslator, OllamaTranslator, XinferenceTranslator, AzureOpenAITranslator,
                           OpenAITranslator, ZhipuTranslator, ModelScopeTranslator, SiliconTranslator, GeminiTranslator, AzureTranslator, TencentTranslator, DifyTranslator, AnythingLLMTranslator, ArgosTranslator, GrokTranslator, GroqTranslator, DeepseekTranslator, OpenAIlikedTranslator, QwenMtTranslator,]:
            if service_name == translator.name:
                self.translator = translator(lang_in, lang_out, service_model, envs=envs, prompt=prompt, ignore_cache=ignore_cache)
        if not self.translator:
            raise ValueError("Unsupported translation service")

    def _raster(self, pageid):
        """Render the ORIGINAL page (2x) so that colours can be read from pixels.
        Returns (array, scale, page_height) or (None, 0, 0) if that is impossible."""
        if self._ras is not None and self._ras[0] == pageid:
            return self._ras[1:]
        try:
            fp = self._pdfdoc._parser.fp
            if self._mdoc is None or self._mdoc_fp is not fp:
                data = fp.getvalue() if hasattr(fp, "getvalue") else (fp.seek(0), fp.read())[1]
                self._mdoc = Document(stream=data)
                self._mdoc_fp = fp
            pg = self._mdoc[pageid]
            if pg.rotation:
                raise ValueError("rotated page")
            sc = 2.0
            pix = pg.get_pixmap(matrix=Matrix(sc, sc), colorspace=csRGB, alpha=False)
            arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            self._ras = (pageid, arr, sc, pix.height / sc)
        except Exception as e:
            log.warning(f"colour sampling not available on page {pageid + 1}: {e!r}")
            self._ras = (pageid, None, 0, 0)
        return self._ras[1:]

    def _sample_color(self, pageid, boxes):
        """Colour of the text inside `boxes` (glyph boxes in PDF points), read from the rendered page.
        Works for any PDF colour space (Separation, DeviceN, Lab, Pattern...)."""
        arr, sc, H = self._raster(pageid)
        if arr is None or not boxes:
            return None
        h, w = arr.shape[:2]
        parts = []
        for x0, y0, x1, y1 in boxes[:400]:
            c0, c1 = max(int(x0 * sc), 0), min(int(x1 * sc) + 1, w)
            r0, r1 = max(int((H - y1) * sc), 0), min(int((H - y0) * sc) + 1, h)
            if c1 > c0 and r1 > r0:
                parts.append(arr[r0:r1, c0:c1].reshape(-1, 3))
        if not parts:
            return None
        reg = np.concatenate(parts).astype(np.int32)
        q = reg >> 4
        keys = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
        bg = reg[keys == np.bincount(keys).argmax()].mean(axis=0)   # background = most common colour
        dist = np.abs(reg - bg).sum(axis=1)
        mx = dist.max()
        if mx < 120:                                                 # nothing visible
            return None
        col = np.median(reg[dist >= 0.8 * mx], axis=0) / 255.0       # the most "inky" pixels = the text
        if col.max() < 0.12:
            return "0 g"
        return "{:.3f} {:.3f} {:.3f} rg".format(*col)

    def receive_layout(self, ltpage: LTPage):
        # 段落
        sstk: list[str] = []            # 段落文字栈
        pstk: list[Paragraph] = []      # 段落属性栈
        prec: list[list] = []           # mỗi đoạn: [(idx, độ dài, lệnh màu|None, khung|None)] của từng ký tự chữ
        pcls: list[int] = []            # lớp layout của từng đoạn
        vbkt: int = 0                   # 段落公式括号计数
        # 公式组
        vstk: list[LTChar] = []         # 公式符号组
        vlstk: list[LTLine] = []        # 公式线条组
        vfix: float = 0                 # 公式纵向偏移
        # 公式组栈
        var: list[list[LTChar]] = []    # 公式符号组栈
        varl: list[list[LTLine]] = []   # 公式线条组栈
        varf: list[float] = []          # 公式纵向偏移栈
        vlen: list[float] = []          # 公式宽度栈
        # 全局
        lstk: list[LTLine] = []         # 全局线条栈
        xt: LTChar = None               # 上一个字符
        xt_cls: int = -1                # 上一个字符所属段落，保证无论第一个字符属于哪个类别都可以触发新段落
        tok_gap: float = 0.0            # 当前行最后一个词之前的水平间距
        tok_start: int = 0              # 当前行最后一个词在 sstk[-1] 中的起始位置
        toc_split = True                # 目录条目（行尾远处的短页码）拆成独立段落
        tok_chars: list[LTChar] = []    # 当前行最后一个词的字符
        tok_x1: float = 0.0             # 最后一个词之前，段落的右边界
        xns: LTChar = None              # 上一个非空格文字字符
        toc_ent: list[int] = []         # 被拆出页码的目录条目（段落序号）
        toc_numx: list[float] = []      # 页码列的左边界
        n_sampled = [0]                 # số từ phải lấy màu từ pixel (chẩn đoán)
        fin_cols: list[str] = []        # 每个段落最终使用的颜色（诊断用）

        def _toc_tail_ok():             # 当前段落是否以“远离正文的短页码”结尾
            if not tok_chars or len(tok_chars) > 8 or not pstk or tok_start <= 0:
                return False
            if tok_gap <= 2.5 * pstk[-1].size:
                return False
            if not sstk[-1][:tok_start].strip():
                return False
            return sstk[-1][tok_start:] == "".join(c.get_text() for c in tok_chars)

        def _toc_cut():                 # 把页码拆成独立段落，保持在原位（右对齐）
            num = list(tok_chars)
            txt = sstk[-1][tok_start:]
            sstk[-1] = sstk[-1][:tok_start].rstrip()
            pstk[-1].x1 = tok_x1        # 条目文字的右边界不含页码
            pstk[-1].brk = True
            toc_ent.append(len(pstk) - 1)
            c0 = num[0]
            toc_numx.append(min(c.x0 for c in num))
            nrecs, off = [], 0          # bản ghi màu của số trang (vị trí tính trong txt)
            for c in num:
                t = c.get_text()
                op = _fill_color_op(c)
                nrecs.append((off, len(t), op, None if op else (c.x0, c.y0, c.x1, c.y1), _style_of(c)))
                off += len(t)
            sstk.append(txt)
            pstk.append(Paragraph(c0.y0, c0.x0, min(c.x0 for c in num), max(c.x1 for c in num),
                                  min(c.y0 for c in num), max(c.y1 for c in num), c0.size, False))
            prec.append(nrecs)
            pcls.append(pcls[-1])
        vmax: float = ltpage.width / 4  # 行内公式最大宽度
        ops: str = ""                   # 渲染结果

        def vflag(font: str, char: str):    # 匹配公式（和角标）字体
            if isinstance(font, bytes):     # 不一定能 decode，直接转 str
                try:
                    font = font.decode('utf-8')  # 尝试使用 UTF-8 解码
                except UnicodeDecodeError:
                    font = ""
            font = font.split("+")[-1]      # 字体名截断
            if re.match(r"\(cid:", char):
                return True
            # ToUnicode CMap của font bị thiếu entry cho một số glyph (hay gặp ở
            # biến thể chữ hoa nhỏ/small-caps hoặc ligature do font bị subset):
            # to_unichr() không raise lỗi mà âm thầm trả về NUL ("\x00") hoặc các
            # control char khác. Đã xác minh trên file thật (vd. font
            # "ACaslonPro-Semibold" trong "The New Map"): các glyph rời rạc, có cid
            # khác nhau (nên vẫn có hình glyph thật trong font), nhưng to_unichr
            # đều ra "\x00". Nếu cứ để sstk nhận NUL rồi gửi đi dịch, ký tự bị mất
            # (dịch máy/encode URL nuốt mất NUL) hoặc bị dịch+vẽ lại bằng Noto
            # thành rác đọc được như "%00%00...". Nên coi các glyph này là công
            # thức -> giữ nguyên, vẽ lại bằng đúng cid + font gốc (bỏ qua dịch),
            # ra đúng hình chữ hoa nhỏ như bản gốc dù không dịch được.
            if char and all((ord(c) < 0x20 or ord(c) == 0x7F) for c in char):
                return True
            # Một số font/công cụ sinh PDF khác lại dùng chuỗi giữ chỗ dạng văn
            # bản "%00", "%1a"... thay vì NUL thật -> xử lý tương tự cho chắc.
            if re.fullmatch(r"(?:%[0-9A-Fa-f]{2})+", char):
                return True
            # 基于字体名规则的判定
            if self.vfont:
                if re.match(self.vfont, font):
                    return True
            else:
                if re.match(                                            # latex 字体
                    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)",
                    font,
                ):
                    return True
            # 基于字符集规则的判定
            if self.vchar:
                if re.match(self.vchar, char):
                    return True
            else:
                if (
                    char
                    and char != " "                                     # 非空格
                    and (
                        unicodedata.category(char[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]   # 文字修饰符、数学符号、分隔符号
                        or ord(char[0]) in range(0x370, 0x400)          # 希腊字母
                    )
                ):
                    return True
            return False

        # 列表/目录类的版面框（各条目之间有明显空行）-> 每个条目单独成段
        try:
            list_cls = spaced_list_boxes(ltpage, self.layout[ltpage.pageid]) if toc_split else set()
            _page_w = self.layout[ltpage.pageid].shape[1]
        except Exception:
            list_cls, _page_w = set(), ltpage.width

        # Tách đoạn/mục theo luật would-fit + treo lề (mục lục tra cứu, danh mục tham khảo,
        # đoạn bị layout model gộp nhầm). hang_cls: các khung kiểu treo lề (dòng nối tiếp thụt vào).
        # Đồng thời tính broken_merge_ids: các ký tự đệm (dấu cách/nháy/gạch nối) nằm giữa
        # hai glyph lỗi, cần được gộp vào chung một khối công thức (xem broken_glyph_merge_ids()).
        try:
            _lay = self.layout[ltpage.pageid]
            _lh, _lw = _lay.shape

            def _cls_of(c):
                return _lay[np.clip(int(c.y0), 0, _lh - 1), np.clip(int(c.x0), 0, _lw - 1)]

            _page_chars = [c for c in ltpage if isinstance(c, LTChar)]
            rf_breaks, hang_cls = (
                reflow_plan(_page_chars, _cls_of)
                if toc_split else (set(), set())
            )
            broken_merge_ids = broken_glyph_merge_ids(_page_chars, _cls_of, vflag)
        except Exception as e:
            log.warning(f"reflow_plan/broken_glyph_merge_ids failed on page {ltpage.pageid + 1}: {e!r}")
            rf_breaks, hang_cls = set(), set()
            broken_merge_ids = set()

        ############################################################
        # A. 原文档解析
        for child in ltpage:
            if isinstance(child, LTChar):
                cur_v = False
                layout = self.layout[ltpage.pageid]
                # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
                h, w = layout.shape
                # 读取当前字符在 layout 中的类别
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                # 锚定文档中 bullet 的位置
                if child.get_text() == "•":
                    cls = 0
                # 判定当前字符是否属于公式
                if (                                                                                        # 判定当前字符是否属于公式
                    cls == 0                                                                                # 1. 类别为保留区域
                    or (cls == xt_cls and len(sstk[-1].strip()) > 1 and child.size < pstk[-1].size * 0.79)  # 2. 角标字体，有 0.76 的角标和 0.799 的大写，这里用 0.79 取中，同时考虑首字母放大的情况
                    or vflag(child.fontname, child.get_text())                                              # 3. 公式字体
                    or (child.matrix[0] == 0 and child.matrix[3] == 0)                                      # 4. 垂直字体
                    or _obj_id(child) in broken_merge_ids                                                    # 5. dấu cách/nháy/gạch nối nằm giữa hai glyph lỗi (gộp lại, xem broken_glyph_merge_ids())
                ):
                    cur_v = True
                # 判定括号组是否属于公式
                if not cur_v:
                    if vstk and child.get_text() == "(":
                        cur_v = True
                        vbkt += 1
                    if vbkt and child.get_text() == ")":
                        cur_v = True
                        vbkt -= 1
                if (                                                        # 判定当前公式是否结束
                    not cur_v                                               # 1. 当前字符不属于公式
                    or cls != xt_cls                                        # 2. 当前字符与前一个字符不属于同一段落
                    # or (abs(child.x0 - xt.x0) > vmax and cls != 0)        # 3. 段落内换行，可能是一长串斜体的段落，也可能是段内分式换行，这里设个阈值进行区分
                    # 禁止纯公式（代码）段落换行，直到文字开始再重开文字段落，保证只存在两种情况
                    # A. 纯公式（代码）段落（锚定绝对位置）sstk[-1]=="" -> sstk[-1]=="{v*}"
                    # B. 文字开头段落（排版相对位置）sstk[-1]!=""
                    or (sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)    # 因为 cls==xt_cls==0 一定有 sstk[-1]==""，所以这里不需要再判定 cls!=0
                ):
                    if vstk:
                        if (                                                # 根据公式右侧的文字修正公式的纵向偏移
                            not cur_v                                       # 1. 当前字符不属于公式
                            and cls == xt_cls                               # 2. 当前字符与前一个字符属于同一段落
                            and child.x0 > max([vch.x0 for vch in vstk])    # 3. 当前字符在公式右侧
                        ):
                            vfix = vstk[0].y0 - child.y0
                        if sstk[-1] == "":
                            xt_cls = -1 # 禁止纯公式段落（sstk[-1]=="{v*}"）的后续连接，但是要考虑新字符和后续字符的连接，所以这里修改的是上个字符的类别
                        sstk[-1] += f"{{v{len(var)}}}"
                        var.append(vstk)
                        varl.append(vlstk)
                        varf.append(vfix)
                        vstk = []
                        vlstk = []
                        vfix = 0
                # 当前字符不属于公式或当前字符是公式的第一个字符
                if not vstk:
                    toc_end = (                     # 上一行以“远离正文的短数字（页码）”结尾 => 目录条目结束，下一行是新条目
                        toc_split
                        and cls == xt_cls
                        and child.x1 < xt.x0
                        and _toc_tail_ok()
                    )
                    gap_end = (                     # 列表/目录：各条目之间有明显空行 => 新条目
                        not toc_end
                        and cls == xt_cls
                        and int(cls) in list_cls
                        and child.x1 < xt.x0
                        and (xns.y0 if xns is not None else xt.y0) - child.y0 > 1.6 * pstk[-1].size
                    )
                    rf_end = (                      # would-fit / treo lề: dòng này mở đầu một đoạn / mục mới
                        not toc_end
                        and not gap_end
                        and cls == xt_cls
                        and _obj_id(child) in rf_breaks
                    )
                    if cls == xt_cls and not toc_end and not gap_end and not rf_end:   # 当前字符与前一个字符属于同一段落
                        if child.x0 > xt.x1 + 1:    # 添加行内空格
                            sstk[-1] += " "
                            tok_gap = child.x0 - (xns.x1 if xns is not None else xt.x1)
                            tok_start = len(sstk[-1])
                            tok_chars = []
                            tok_x1 = pstk[-1].x1
                        elif child.x1 < xt.x0:      # 添加换行空格并标记原文段落存在换行
                            sstk[-1] += " "
                            pstk[-1].brk = True
                            if pstk[-1].cx is None and child.get_text() != " ":
                                pstk[-1].cx = child.x0      # nhớ x của dòng nối tiếp đầu tiên (treo lề)
                            tok_gap = 0.0
                            tok_start = len(sstk[-1])
                            tok_chars = []
                            tok_x1 = pstk[-1].x1
                        elif (xt.get_text() == " " and child.get_text() != " "
                              and xns is not None and child.x0 > xns.x1 + 1):   # 空格字符后的新词
                            tok_gap = child.x0 - xns.x1
                            tok_start = len(sstk[-1])
                            tok_chars = []
                            tok_x1 = pstk[-1].x1
                    else:                           # 根据当前字符构建一个新的段落
                        if toc_end:
                            _toc_cut()
                        sstk.append("")
                        pstk.append(Paragraph(child.y0, child.x0, child.x0, child.x0, child.y0, child.y1, child.size, False))
                        prec.append([])
                        pcls.append(int(cls))
                        tok_gap = 0.0
                        tok_start = 0
                        tok_chars = []
                        tok_x1 = 0.0
                        xns = None
                if not cur_v:                                               # 文字入栈
                    if (                                                    # 根据当前字符修正段落属性
                        child.size > pstk[-1].size                          # 1. 当前字符比段落字体大
                        or len(sstk[-1].strip()) == 1                       # 2. 当前字符为段落第二个文字（考虑首字母放大的情况）
                    ) and child.get_text() != " ":                          # 3. 当前字符不是空格
                        pstk[-1].y -= child.size - pstk[-1].size            # 修正段落初始纵坐标，假设两个不同大小字符的上边界对齐
                        pstk[-1].size = child.size
                    sstk[-1] += child.get_text()
                    if child.get_text() != " ":
                        _c = _fill_color_op(child)
                        _t = child.get_text()
                        # ghi màu từng ký tự (None = không đọc được từ lệnh -> sẽ lấy mẫu pixel theo từ)
                        prec[-1].append((len(sstk[-1]) - len(_t), len(_t), _c,
                                         None if _c else (child.x0, child.y0, child.x1, child.y1),
                                         _style_of(child)))
                        tok_chars.append(child)
                        xns = child
                else:                                                       # 公式入栈
                    if (                                                    # 根据公式左侧的文字修正公式的纵向偏移
                        not vstk                                            # 1. 当前字符是公式的第一个字符
                        and cls == xt_cls                                   # 2. 当前字符与前一个字符属于同一段落
                        and child.x0 > xt.x0                                # 3. 前一个字符在公式左侧
                    ):
                        vfix = child.y0 - xt.y0
                    vstk.append(child)
                # 更新段落边界，因为段落内换行之后可能是公式开头，所以要在外边处理
                pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                pstk[-1].y0 = min(pstk[-1].y0, child.y0)
                pstk[-1].y1 = max(pstk[-1].y1, child.y1)
                # 更新上一个字符
                xt = child
                xt_cls = cls
            elif isinstance(child, LTFigure):   # 图表
                pass
            elif isinstance(child, LTLine):     # 线条
                layout = self.layout[ltpage.pageid]
                # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
                h, w = layout.shape
                # 读取当前线条在 layout 中的类别
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                if vstk and cls == xt_cls:      # 公式线条
                    vlstk.append(child)
                else:                           # 全局线条
                    lstk.append(child)
            else:
                pass
        # 处理结尾
        if toc_split and toc_ent and not vstk and _toc_tail_ok():
            _toc_cut()
        if toc_ent:                         # 同一页的目录条目使用同一个右边界
            _col_r = max(max(pstk[i].x1 for i in toc_ent),                  # 译文可以用到页码列之前
                         min(toc_numx) - 1.5 * pstk[toc_ent[0]].size)
            for i in toc_ent:
                pstk[i].x1 = _col_r
        if vstk:    # 公式出栈
            sstk[-1] += f"{{v{len(var)}}}"
            var.append(vstk)
            varl.append(vlstk)
            varf.append(vfix)
        log.debug("\n==========[VSTACK]==========\n")
        for id, v in enumerate(var):  # 计算公式宽度
            l = max([vch.x1 for vch in v]) - v[0].x0
            log.debug(f'< {l:.1f} {v[0].x0:.1f} {v[0].y0:.1f} {v[0].cid} {v[0].fontname} {len(varl[id])} > v{id} = {"".join([ch.get_text() for ch in v])}')
            vlen.append(l)

        ############################################################
        # Màu theo từng đoạn chữ: chia mỗi đoạn thành các run cùng màu.
        # plans[i] = (màu nền của đoạn, [(start, end, màu) các run khác màu nền])
        def _samp(boxes):
            n_sampled[0] += 1
            return self._sample_color(ltpage.pageid, boxes)

        plans = []
        for i in range(len(sstk)):
            try:
                segs = make_segments(sstk[i], fill_unknown(prec[i], _samp), pick_op)
                plans.append(plan_runs(segs, sstk[i]))
            except Exception as e:
                log.warning(f"colour runs failed on page {ltpage.pageid + 1}: {e!r}")
                plans.append((None, []))

        ############################################################
        # B. 段落翻译
        log.debug("\n==========[SSTACK]==========\n")

        # Dọn placeholder "%00"/"%1a"... (ToUnicode CMap hỏng, xem vflag()) và các
        # ký tự điều khiển (NUL...) có thể lẫn trong văn bản gốc trước khi gửi đi
        # dịch/cache: nếu lọt qua, chúng có thể bị URL-encode (vd. NUL -> "%00")
        # hoặc dịch máy trả về nguyên văn, khiến bản dịch hiện rác trên trang.
        _CID_PLACEHOLDER_RE = re.compile(r"(?:%[0-9A-Fa-f]{2})+")
        _CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

        def _sanitize(s: str) -> str:
            s = _CID_PLACEHOLDER_RE.sub("", s)
            s = _CTRL_CHAR_RE.sub("", s)
            return s

        @retry(wait=wait_fixed(1))
        def worker(s: str):  # 多线程翻译
            s = _sanitize(s)
            if not s.strip() or re.match(r"^\{v\d+\}$", s) or re.fullmatch(r"[\d\s.,:;()\-–]+", s):  # 空白、公式、纯数字不翻译
                return s
            try:
                new = self.translator.translate(s)
                return _sanitize(new)
            except BaseException as e:
                if log.isEnabledFor(logging.DEBUG):
                    log.exception(e)
                else:
                    log.exception(e, exc_info=False)
                raise e
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, self.thread)
        ) as executor:
            news = list(executor.map(worker, sstk))

            # Các run khác màu có chữ cái mà bản dịch của cả đoạn không chứa nguyên văn
            # (vd. "Fig. 10.6" -> "Hình 10.6") được dịch riêng để định vị trong bản dịch.
            _extra = []
            for i, new in enumerate(news):
                for s, e, op in plans[i][1]:
                    t = sstk[i][s:e].strip()
                    if t and t not in _extra and needs_translation(t) and find_exact(new, t) is None:
                        _extra.append(t)
            tr = dict(zip(_extra, executor.map(worker, _extra))) if _extra else {}

        cmaps = []          # màu từng ký tự của bản dịch
        n_runs = 0
        for i, new in enumerate(news):
            _base, _runs = plans[i]
            _base = _base or ("0 g", 0)
            try:
                _spans = locate_spans(new, sstk[i], _runs, tr) if _runs else []
                cmaps.append(colour_map(new, _base, _spans))
                n_runs += len(_spans)
            except Exception as e:
                log.warning(f"colour mapping failed on page {ltpage.pageid + 1}: {e!r}")
                cmaps.append([_base] * len(new))

        ############################################################
        # C. 新文档排版
        def raw_string(fcur: str, cstk: str):  # 编码字符串
            if fcur == self.noto_name:
                return "".join(["%04x" % self.noto.has_glyph(ord(c)) for c in cstk])
            elif isinstance(self.fontmap[fcur], PDFCIDFont):  # 判断编码长度
                return "".join(["%04x" % ord(c) for c in cstk])
            else:
                return "".join(["%02x" % ord(c) for c in cstk])

        # 根据目标语言获取默认行距
        LANG_LINEHEIGHT_MAP = {
            "zh-cn": 1.4, "zh-tw": 1.4, "zh-hans": 1.4, "zh-hant": 1.4, "zh": 1.4,
            "ja": 1.1, "ko": 1.2, "en": 1.2, "ar": 1.0, "ru": 0.8, "uk": 0.8, "ta": 0.8
        }
        default_line_height = LANG_LINEHEIGHT_MAP.get(self.translator.lang_out.lower(), 1.1) # 小语种默认1.1
        _x, _y = 0, 0
        ops_list = []

        # ---- Layout tweaks for non-CJK target languages (e.g. Vietnamese) ----
        _lang_out = self.translator.lang_out.lower()
        _cjk = _lang_out.split("-")[0] in ("zh", "ja", "ko")
        wrap_by_word = not _cjk  # wrap at spaces instead of at any character
        fit_shrink = not _cjk  # shrink font a little when the translation is longer
        # "tiro" (Times) has no Vietnamese/Polish/Czech... letters, which makes words mix two fonts
        _latin1_ok = {"en", "fr", "de", "es", "it", "pt", "nl", "da", "sv", "no", "nb", "nn", "fi", "ca", "gl", "id", "ms", "af", "sw", "eu"}
        force_noto = (not _cjk) and (_lang_out.split("-")[0] not in _latin1_ok)

        def _char_adv(c, fsize):
            try:
                if not force_noto and self.fontmap["tiro"].to_unichr(ord(c)) == c:
                    return self.fontmap["tiro"].char_width(ord(c)) * fsize
            except Exception:
                pass
            return self.noto.char_lengths(c, fsize)[0]

        def _est_total_width(text, fsize):
            total = 0.0
            i = 0
            while i < len(text):
                m = re.match(r"\{\s*v([\d\s]+)\}", text[i:], re.IGNORECASE)
                if m:
                    try:
                        total += vlen[int(m.group(1).replace(" ", ""))]
                    except Exception:
                        pass
                    i += len(m.group(0))
                else:
                    total += _char_adv(text[i], fsize)
                    i += 1
            return total

        def _stroke_op(col):                        # "r g b rg" -> "r g b RG"
            return col[:-2] + col[-2:].replace("rg", "RG").replace("g", "G").replace("k", "K") if col else "0 G"

        def gen_op_txt(font, size, x, y, rtxt, color=None):
            col, sty = color if isinstance(color, tuple) else (color, 0)
            col = col or "0 g"
            skew = 0.21 if sty & 2 else 0.0         # nghiêng ~12 độ
            pre = post = ""
            if sty & 1:                             # đậm: tô + viền cùng màu chữ
                pre = f"{_stroke_op(col)} 1 j 2 Tr {max(size * 0.03, 0.05):f} w "
                post = "0 Tr 0 j 0 G "
            return f"{col} {pre}/{font} {size:f} Tf 1 0 {skew:f} 1 {x:f} {y:f} Tm [<{rtxt}>] TJ {post}"

        def gen_op_line(x, y, xlen, ylen, linewidth):
            return f"ET q 1 0 0 1 {x:f} {y:f} cm [] 0 d 0 J {linewidth:f} w 0 0 m {xlen:f} {ylen:f} l S Q BT "

        _vcol_cache: dict = {}

        def _vcolor(vch):                           # màu ký tự công thức: từ lệnh, hoặc lấy mẫu pixel
            c = _fill_color_op(vch)
            if c is None:
                k = _obj_id(vch)                    # `id` bị biến vòng lặp `for id, new in ...` che mất
                if k not in _vcol_cache:
                    _vcol_cache[k] = self._sample_color(ltpage.pageid, [(vch.x0, vch.y0, vch.x1, vch.y1)])
                c = _vcol_cache[k]
            return c

        for id, new in enumerate(news):
            x: float = pstk[id].x                       # 段落初始横坐标
            y: float = pstk[id].y                       # 段落初始纵坐标
            x0: float = pstk[id].x0                     # 段落左边界
            x1: float = pstk[id].x1                     # 段落右边界
            height: float = pstk[id].y1 - pstk[id].y0   # 段落高度
            size: float = pstk[id].size                 # 段落字体大小
            pcolor: tuple = plans[id][0] or ("0 g", 0)  # (màu, style) nền của đoạn: của phần chữ chiếm đa số
            cmap: list = cmaps[id]                      # màu từng ký tự của bản dịch
            fin_cols.append(pcolor[0])
            brk: bool = pstk[id].brk                    # 段落换行标记
            cstk: str = ""                              # 当前文字栈
            fcur: str = None                            # 当前字体 ID
            colcur: str = None                          # màu của cstk
            lidx = 0                                    # 记录换行次数
            tx = x
            fcur_ = fcur
            ptr = 0
            log.debug(f"< {y} {x} {x0} {x1} {size} {brk} > {sstk[id]} | {new}")

            ops_vals: list[dict] = []
            line_x0 = x                                 # start x of the current wrapped line

            # Danh sách treo lề (mục lục tra cứu, tài liệu tham khảo): dòng nối tiếp thụt vào
            # như bản gốc thay vì quay về sát lề trái.
            _hang = (brk and pcls[id] in hang_cls and pstk[id].cx is not None
                     and x0 < pstk[id].cx < x1 - 2 * size)
            cont_x = pstk[id].cx if _hang else x0       # x bắt đầu của các dòng nối tiếp

            if fit_shrink and brk and (x1 - cont_x) > 0 and height > 0:
                _lines = _est_total_width(new, size) * 1.06 / (x1 - cont_x)
                _need = (_lines + 0.5) * size
                if _need > height:
                    size *= max((height / _need) ** 0.5, 0.8)
            elif fit_shrink and not brk and new.strip():
                # single-line paragraph (heading, TOC entry, caption, drop cap...):
                # shrink so that the translation does not run over its right neighbour
                _room = free_right(pstk, id, _page_w) - x
                _need_w = _est_total_width(new, size)
                if _room > 0 and _need_w > _room:
                    size *= max(_room / _need_w, 0.5)

            while ptr < len(new):
                force_wrap = False
                ccur = cmap[ptr]                        # màu của ký tự đang xử lý
                vy_regex = re.match(
                    r"\{\s*v([\d\s]+)\}", new[ptr:], re.IGNORECASE
                )  # 匹配 {vn} 公式标记
                mod = 0  # 文字修饰符
                if vy_regex:  # 加载公式
                    ptr += len(vy_regex.group(0))
                    try:
                        vid = int(vy_regex.group(1).replace(" ", ""))
                        adv = vlen[vid]
                    except Exception:
                        continue  # 翻译器可能会自动补个越界的公式标记
                    if var[vid][-1].get_text() and unicodedata.category(var[vid][-1].get_text()[0]) in ["Lm", "Mn", "Sk"]:  # 文字修饰符
                        mod = var[vid][-1].width
                else:  # 加载文字
                    ch = new[ptr]
                    fcur_ = None
                    try:
                        if fcur_ is None and not force_noto and self.fontmap["tiro"].to_unichr(ord(ch)) == ch:
                            fcur_ = "tiro"  # 默认拉丁字体
                    except Exception:
                        pass
                    if fcur_ is None:
                        fcur_ = self.noto_name  # 默认非拉丁字体
                    if fcur_ == self.noto_name: # FIXME: change to CONST
                        adv = self.noto.char_lengths(ch, size)[0]
                    else:
                        adv = self.fontmap[fcur_].char_width(ord(ch)) * size
                    ptr += 1
                    if wrap_by_word and brk and ch != " " and (ptr == 1 or new[ptr - 2] == " "):
                        # start of a word: if the whole word does not fit, wrap before it
                        _wlen = adv
                        _j = ptr
                        while _j < len(new) and new[_j] not in " {":
                            _wlen += _char_adv(new[_j], size)
                            _j += 1
                        if x > line_x0 + 0.01 and x + _wlen > x1 + 0.1 * size and _wlen <= (x1 - line_x0):
                            force_wrap = True
                if (                                # 输出文字缓冲区
                    fcur_ != fcur                   # 1. 字体更新
                    or vy_regex                     # 2. 插入公式
                    or x + adv > x1 + 0.1 * size    # 3. 到达右边界（可能一整行都被符号化，这里需要考虑浮点误差）
                    or force_wrap                   # 4. 单词放不下，整词换行
                    or ccur != colcur               # 5. đổi màu
                ):
                    if cstk:
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": fcur,
                            "size": size,
                            "x": tx,
                            "dy": 0,
                            "rtxt": raw_string(fcur, cstk),
                            "color": colcur or pcolor,
                            "lidx": lidx
                        })
                        cstk = ""
                if brk and (x + adv > x1 + 0.1 * size or force_wrap):  # 到达右边界且原文段落存在换行
                    lidx += 1
                    # a wrapped line starts at the left margin (or at the hanging indent), unless a drop cap / floated item sits there
                    x = left_float_end(pstk, id, y - lidx * size * default_line_height, size, cont_x)
                    line_x0 = x
                if vy_regex:  # 插入公式
                    fix = 0
                    if fcur is not None:  # 段落内公式修正纵向偏移
                        fix = varf[vid]
                    for vch in var[vid]:  # 排版公式字符
                        vc = chr(vch.cid)
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": self.fontid[vch.font],
                            "size": vch.size,
                            "x": x + vch.x0 - var[vid][0].x0,
                            "dy": fix + vch.y0 - var[vid][0].y0,
                            "rtxt": raw_string(self.fontid[vch.font], vc),
                            "color": _vcolor(vch) or (ccur or pcolor)[0],
                            "lidx": lidx
                        })
                        if log.isEnabledFor(logging.DEBUG):
                            lstk.append(LTLine(0.1, (_x, _y), (x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0)))
                            _x, _y = x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0
                    for l in varl[vid]:  # 排版公式线条
                        if l.linewidth < 5:  # hack 有的文档会用粗线条当图片背景
                            ops_vals.append({
                                "type": OpType.LINE,
                                "x": l.pts[0][0] + x - var[vid][0].x0,
                                "dy": l.pts[0][1] + fix - var[vid][0].y0,
                                "linewidth": l.linewidth,
                                "xlen": l.pts[1][0] - l.pts[0][0],
                                "ylen": l.pts[1][1] - l.pts[0][1],
                                "lidx": lidx
                            })
                else:  # 插入文字缓冲区
                    if not cstk:  # 单行开头
                        tx = x
                        if x == line_x0 and ch == " ":  # 消除段落换行空格
                            adv = 0
                        else:
                            cstk += ch
                            colcur = ccur
                    else:
                        cstk += ch
                adv -= mod # 文字修饰符
                fcur = fcur_
                x += adv
                if log.isEnabledFor(logging.DEBUG):
                    lstk.append(LTLine(0.1, (_x, _y), (x, y)))
                    _x, _y = x, y
            # 处理结尾
            if cstk:
                ops_vals.append({
                    "type": OpType.TEXT,
                    "font": fcur,
                    "size": size,
                    "x": tx,
                    "dy": 0,
                    "rtxt": raw_string(fcur, cstk),
                    "color": colcur or pcolor,
                    "lidx": lidx
                })

            line_height = default_line_height

            while (lidx + 1) * size * line_height > height and line_height >= 1:
                line_height -= 0.05

            for vals in ops_vals:
                if vals["type"] == OpType.TEXT:
                    ops_list.append(gen_op_txt(vals["font"], vals["size"], vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["rtxt"], vals.get("color")))
                elif vals["type"] == OpType.LINE:
                    ops_list.append(gen_op_line(vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["xlen"], vals["ylen"], vals["linewidth"]))

        for l in lstk:  # 排版全局线条
            if l.linewidth < 5:  # hack 有的文档会用粗线条当图片背景
                ops_list.append(gen_op_line(l.pts[0][0], l.pts[0][1], l.pts[1][0] - l.pts[0][0], l.pts[1][1] - l.pts[0][1], l.linewidth))

        if _DIAG:
            _cols = Counter(fin_cols)
            print(f"[pdf2zh-fix] page {ltpage.pageid + 1}: paragraphs={len(news)} toc_entries={len(toc_ent)} "
                  f"list_boxes={len(list_cls)} reflow_breaks={len(rf_breaks)} hanging_boxes={len(hang_cls)} "
                  f"broken_merge_fillers={len(broken_merge_ids)} "
                  f"colour_runs={n_runs} colour_from_pixels={n_sampled[0]} colours={dict(_cols)}")
        ops = f"q BT {''.join(ops_list)}ET Q "
        return ops


class OpType(Enum):
    TEXT = "text"
    LINE = "line"
    