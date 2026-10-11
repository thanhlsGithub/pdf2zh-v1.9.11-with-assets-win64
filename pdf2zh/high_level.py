"""Functions that can be used for the most common use-cases for pdf2zh.six"""

import asyncio
import io
import os
import re
import sys
import tempfile
import logging
from asyncio import CancelledError
from pathlib import Path
from string import Template
from typing import Any, BinaryIO, List, Optional, Dict

import numpy as np
import requests
import tqdm
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pymupdf import Document, Font

from pdf2zh.converter import TranslateConverter
from pdf2zh.doclayout import OnnxModel
from pdf2zh.pdfinterp import PDFPageInterpreterEx

from pdf2zh.config import ConfigManager
from pdf2zh.logsetup import set_progress, stage, fmt_pages
from babeldoc.assets.assets import get_font_and_metadata

NOTO_NAME = "noto"

logger = logging.getLogger(__name__)

# True  -> output files are named by language:
#            <name>-VI.pdf     (translated only, "mono")
#            <name>-EN-VI.pdf  (bilingual, "dual": source language first, then target)
# False -> old names: <name>-mono.pdf / <name>-dual.pdf
USE_LANG_SUFFIX = True

# Language codes whose file suffix is not simply the upper-cased code.
# (Vietnamese used to be forced to "VN" here; that override has been
# removed so it now uses the ISO 639-1 code "VI" via the .upper()
# fallback in lang_suffix() below, like every other language.)
_LANG_SUFFIX_MAP = {}

OUTPUT_FORMATS = ("mono", "dual", "both")


def lang_suffix(code: str) -> str:
    """'vi' -> 'VI', 'en' -> 'EN', 'ja' -> 'JA', 'fr' -> 'FR', 'de' -> 'DE'..."""
    code = (code or "").strip()
    return _LANG_SUFFIX_MAP.get(code.lower(), code.upper())


def get_output_paths(output, filename: str, lang_in: str, lang_out: str, suffix: str = ""):
    """Return (mono_path, dual_path) for a translated document.

    `suffix` is appended right before the file extension (e.g. "-BabelDOC"),
    so different translation engines can share the same naming scheme while
    still producing distinguishable file names.
    """
    out = lang_suffix(lang_out)
    if not USE_LANG_SUFFIX or not out:
        mono_name = f"{filename}-mono{suffix}.pdf"
        dual_name = f"{filename}-dual{suffix}.pdf"
    else:
        inn = lang_suffix(lang_in)
        mono_name = f"{filename}-{out}{suffix}.pdf"
        dual_name = f"{filename}-{inn}-{out}{suffix}.pdf" if inn else f"{filename}-dual-{out}{suffix}.pdf"
    return Path(output) / mono_name, Path(output) / dual_name


noto_list = [
    "am",  # Amharic
    "ar",  # Arabic
    "bn",  # Bengali
    "bg",  # Bulgarian
    "chr",  # Cherokee
    "el",  # Greek
    "gu",  # Gujarati
    "iw",  # Hebrew
    "hi",  # Hindi
    "kn",  # Kannada
    "ml",  # Malayalam
    "mr",  # Marathi
    "ru",  # Russian
    "sr",  # Serbian
    "ta",  # Tamil
    "te",  # Telugu
    "th",  # Thai
    "ur",  # Urdu
    "uk",  # Ukrainian
]


def check_files(files: List[str]) -> List[str]:
    files = [
        f for f in files if not f.startswith("http://")
    ]  # exclude online files, http
    files = [
        f for f in files if not f.startswith("https://")
    ]  # exclude online files, https
    missing_files = [file for file in files if not os.path.exists(file)]
    return missing_files


def translate_patch(
    inf: BinaryIO,
    pages: Optional[list[int]] = None,
    vfont: str = "",
    vchar: str = "",
    thread: int = 0,
    doc_zh: Document = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    noto_name: str = "",
    noto: Font = None,
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    ignore_cache: bool = False,
    ocr_pages: Optional[set] = None,
    **kwarg: Any,
) -> None:
    rsrcmgr = PDFResourceManager()
    layout = {}
    device = TranslateConverter(
        rsrcmgr,
        vfont,
        vchar,
        thread,
        layout,
        lang_in,
        lang_out,
        service,
        noto_name,
        noto,
        envs,
        prompt,
        ignore_cache,
    )

    assert device is not None
    obj_patch = {}
    interpreter = PDFPageInterpreterEx(rsrcmgr, device, obj_patch)
    if pages:
        total_pages = len(pages)
    else:
        total_pages = doc_zh.page_count

    parser = PDFParser(inf)
    doc = PDFDocument(parser)
    with tqdm.tqdm(total=total_pages) as progress:
        for pageno, page in enumerate(PDFPage.create_pages(doc)):
            if cancellation_event and cancellation_event.is_set():
                raise CancelledError("task cancelled")
            if pages and (pageno not in pages):
                continue
            progress.update()
            if callback:
                callback(progress)
            page.pageno = pageno
            set_progress(f"translate page {pageno + 1}/{doc_zh.page_count}")  # chỉ giữ trang mới nhất, không log từng trang
            pix = doc_zh[page.pageno].get_pixmap()
            image = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, 3
            )[:, :, ::-1]
            page_layout = model.predict(image, imgsz=int(pix.height / 32) * 32)[0]
            if ocr_pages and pageno in ocr_pages:
                # image-only page that got an OCR text layer: translate it with
                # pdf2zh/ocr_prep.py (legacy mode) and skip the normal pdfminer path for this page
                from pdf2zh.ocr_prep import translate_ocr_page

                translate_ocr_page(
                    doc_zh[pageno],
                    page_layout,
                    device.translator,
                    noto,
                    thread,
                    cancellation_event,
                )
                continue
            # kdtree 是不可能 kdtree 的，不如直接渲染成图片，用空间换时间
            box = np.ones((pix.height, pix.width))
            h, w = box.shape
            vcls = ["abandon", "figure", "table", "isolate_formula", "formula_caption"]
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] not in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = i + 2
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = 0
            layout[page.pageno] = box
            # 新建一个 xref 存放新指令流
            page.page_xref = doc_zh.get_new_xref()  # hack 插入页面的新 xref
            doc_zh.update_object(page.page_xref, "<<>>")
            doc_zh.update_stream(page.page_xref, b"")
            doc_zh[page.pageno].set_contents(page.page_xref)
            interpreter.process_page(page)

    device.close()
    return obj_patch


# ---------------------------------------------------------------------------
# OCR for image-only pages (ported from upstream pdf2zh main, see pdf2zh/ocr.py)
# ---------------------------------------------------------------------------
# pdf2zh language code -> Tesseract code. Env var PDF2ZH_OCR_LANGUAGE (for
# example "vie" or "eng+vie") overrides this. Anything not listed is passed
# to Tesseract unchanged.
_OCR_LANG_MAP = {
    "en": "eng",
    "zh": "chi_sim",
    "zh-cn": "chi_sim",
    "zh-tw": "chi_tra",
    "ja": "jpn",
    "ko": "kor",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "vi": "vie",  # not in upstream: Tesseract uses "vie", not "vi"
}


def _ocr_tessdata(language):
    """Use TESSDATA_PREFIX if set, otherwise download the language data once."""
    if not re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", language):
        raise ValueError("Invalid OCR language; use Tesseract codes such as eng+vie")
    if tessdata := os.environ.get("TESSDATA_PREFIX"):
        return tessdata
    try:
        import pooch
    except ImportError as exc:
        raise RuntimeError(
            "OCR needs language data. Either set TESSDATA_PREFIX to a folder "
            "containing <lang>.traineddata (e.g. eng.traineddata), or install pooch."
        ) from exc
    cache = Path.home() / ".cache/pdf2zh/tessdata/4.1.0"
    for code in dict.fromkeys(language.split("+")):
        filename = f"{code}.traineddata"
        try:
            pooch.retrieve(
                url=f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/{filename}",
                known_hash=None,
                fname=filename,
                path=cache,
                downloader=pooch.HTTPDownloader(timeout=60),
            )
        except (requests.RequestException, OSError) as exc:
            raise RuntimeError(
                f"Could not download OCR data for '{code}'. Check the language code "
                "and network connection, or set TESSDATA_PREFIX to local data."
            ) from exc
    return str(cache)


def _ocr_pages(doc, pages, lang_in, cancellation_event):
    """Add an invisible text layer to image-only pages of the working copy.

    Returns the set of page numbers that were OCR'd. Pages that already have
    text, or that have no image, are left alone.
    """
    language = os.environ.get("PDF2ZH_OCR_LANGUAGE") or _OCR_LANG_MAP.get(
        (lang_in or "").lower(), lang_in or "eng"
    )
    ocr_pages = set()
    empty = []
    tessdata = None
    for pageno in range(len(doc)):
        if cancellation_event and cancellation_event.is_set():
            raise CancelledError("task cancelled")
        if pages and pageno not in pages:
            continue
        page = doc[pageno]
        if page.get_text().strip() or not page.get_image_info():
            continue
        set_progress(f"OCR page {pageno + 1}/{len(doc)}")
        if tessdata is None:
            tessdata = _ocr_tessdata(language)
        try:
            data = page.get_pixmap(dpi=300, alpha=False, annots=False).pdfocr_tobytes(
                language=language,
                tessdata=tessdata,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"OCR failed on page {pageno + 1} using '{language}' data in {tessdata}. "
                "Check the language data and any TESSDATA_PREFIX override. "
                "Use PDF2ZH_OCR_LANGUAGE to override the OCR language."
            ) from exc
        with Document(stream=data) as recognized:
            if not recognized[0].get_text().strip():
                empty.append(pageno)
                continue
            contents = doc.get_new_xref()
            doc.update_object(contents, "<<>>")
            doc.update_stream(contents, b"")
            page.set_contents(contents)
            page.show_pdf_page(
                page.rect * page.derotation_matrix,
                recognized,
                0,
                rotate=page.rotation,
            )
        ocr_pages.add(pageno)
    if empty:
        logger.warning("OCR found no text on %s", fmt_pages(empty))
    return ocr_pages


def _prepare_ocr(doc, pages, lang_in, cancellation_event, model=None):
    """Chon cach xu ly trang thieu chu (scan / anh / duong cong / trang hon hop).

    PDF2ZH_OCR_MODE=region (mac dinh): pdf2zh/ocr_prep.py, OCR theo VUNG tren moi trang,
        ve lai thanh chu that, trang di tiep pipeline co dien nhu PDF thuong.
    PDF2ZH_OCR_MODE=legacy: cach cu (_ocr_pages + ocr.py), chi OCR trang khong co chu nao.
    """
    if os.environ.get("PDF2ZH_OCR_MODE", "region").strip().lower() == "legacy":
        return _ocr_pages(doc, pages, lang_in, cancellation_event)
    from pdf2zh.ocr_prep import prep_document

    language = os.environ.get("PDF2ZH_OCR_LANGUAGE") or _OCR_LANG_MAP.get(
        (lang_in or "").lower(), lang_in or "eng"
    )
    return prep_document(
        doc,
        pages,
        language,
        lambda: _ocr_tessdata(language),
        cancellation_event,
        model,
    )


def translate_stream(
    stream: bytes,
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    output_format: str = "both",
    **kwarg: Any,
):
    """Translate a PDF given as bytes.

    Returns (mono_bytes, dual_bytes). Depending on `output_format`
    ("mono" / "dual" / "both") the one that is not requested is NOT built and
    is returned as None (this saves the page merging, font subsetting and
    writing time of that file).
    """
    if output_format not in OUTPUT_FORMATS:
        output_format = "both"
    want_mono = output_format in ("mono", "both")
    want_dual = output_format in ("dual", "both")

    font_list = [("tiro", None)]

    font_path = download_remote_fonts(lang_out.lower())
    noto_name = NOTO_NAME
    noto = Font(noto_name, font_path)
    font_list.append((noto_name, font_path))

    doc_en = Document(stream=stream)
    stream = io.BytesIO()
    doc_en.save(stream)
    doc_zh = Document(stream=stream)
    logger.info(
        "translate_stream: %d bytes, pages=%s, %s -> %s, service=%s, thread=%s, format=%s",
        stream.getbuffer().nbytes if hasattr(stream, "getbuffer") else -1,
        fmt_pages(pages), lang_in, lang_out, service, thread, output_format,
    )
    with stage("OCR prep"):
        ocr_pages = _prepare_ocr(doc_zh, pages, lang_in, cancellation_event, model)
    page_count = doc_zh.page_count
    # font_list = [("GoNotoKurrent-Regular.ttf", font_path), ("tiro", None)]
    font_id = {}
    for page in doc_zh:
        for font in font_list:
            font_id[font[0]] = page.insert_font(font[0], font[1])
    xreflen = doc_zh.xref_length()
    for xref in range(1, xreflen):
        for label in ["Resources/", ""]:  # 可能是基于 xobj 的 res
            try:  # xref 读写可能出错
                font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                target_key_prefix = f"{label}Font/"
                if font_res[0] == "xref":
                    resource_xref_id = re.search("(\\d+) 0 R", font_res[1]).group(1)
                    xref = int(resource_xref_id)
                    font_res = ("dict", doc_zh.xref_object(xref))
                    target_key_prefix = ""

                if font_res[0] == "dict":
                    for font in font_list:
                        target_key = f"{target_key_prefix}{font[0]}"
                        font_exist = doc_zh.xref_get_key(xref, target_key)
                        if font_exist[0] == "null":
                            doc_zh.xref_set_key(
                                xref,
                                target_key,
                                f"{font_id[font[0]]} 0 R",
                            )
            except Exception:
                pass

    fp = io.BytesIO()

    doc_zh.save(fp)
    with stage("translate pages"):
        obj_patch: dict = translate_patch(fp, **locals())

    for obj_id, ops_new in obj_patch.items():
        # ops_old=doc_en.xref_stream(obj_id)
        # print(obj_id)
        # print(ops_old)
        # print(ops_new.encode())
        doc_zh.update_stream(obj_id, ops_new.encode())

    if want_dual:  # build the bilingual (side by side) document
        doc_en.insert_file(doc_zh)
        for id in range(page_count):
            doc_en.move_page(page_count + id, id * 2 + 1)
    if not skip_subset_fonts:
        if want_mono:
            doc_zh.subset_fonts(fallback=True)
        if want_dual:
            doc_en.subset_fonts(fallback=True)
    s_mono = doc_zh.write(deflate=True, garbage=3, use_objstms=1) if want_mono else None
    s_dual = doc_en.write(deflate=True, garbage=3, use_objstms=1) if want_dual else None
    return s_mono, s_dual


def convert_to_pdfa(input_path, output_path):
    """
    Convert PDF to PDF/A format

    Args:
        input_path: Path to source PDF file
        output_path: Path to save PDF/A file
    """
    from pikepdf import Dictionary, Name, Pdf

    # Open the PDF file
    pdf = Pdf.open(input_path)

    # Add PDF/A conformance metadata
    metadata = {
        "pdfa_part": "2",
        "pdfa_conformance": "B",
        "title": pdf.docinfo.get("/Title", ""),
        "author": pdf.docinfo.get("/Author", ""),
        "creator": "PDF Math Translate",
    }

    with pdf.open_metadata() as meta:
        meta.load_from_docinfo(pdf.docinfo)
        meta["pdfaid:part"] = metadata["pdfa_part"]
        meta["pdfaid:conformance"] = metadata["pdfa_conformance"]

    # Create OutputIntent dictionary
    output_intent = Dictionary(
        {
            "/Type": Name("/OutputIntent"),
            "/S": Name("/GTS_PDFA1"),
            "/OutputConditionIdentifier": "sRGB IEC61966-2.1",
            "/RegistryName": "http://www.color.org",
            "/Info": "sRGB IEC61966-2.1",
        }
    )

    # Add output intent to PDF root
    if "/OutputIntents" not in pdf.Root:
        pdf.Root.OutputIntents = [output_intent]
    else:
        pdf.Root.OutputIntents.append(output_intent)

    # Save as PDF/A
    pdf.save(output_path, linearize=True)
    pdf.close()


def translate(
    files: list[str],
    output: str = "",
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    compatible: bool = False,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    output_format: str = "both",
    **kwarg: Any,
):
    if not files:
        raise PDFValueError("No files to process.")

    missing_files = check_files(files)

    if missing_files:
        print("The following files do not exist:", file=sys.stderr)
        for file in missing_files:
            print(f"  {file}", file=sys.stderr)
        raise PDFValueError("Some files do not exist.")

    if output_format not in OUTPUT_FORMATS:
        output_format = "both"

    result_files = []

    for file in files:
        if type(file) is str and (
            file.startswith("http://") or file.startswith("https://")
        ):
            print("Online files detected, downloading...")
            try:
                r = requests.get(file, allow_redirects=True)
                if r.status_code == 200:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp_file:
                        print(f"Writing the file: {file}...")
                        tmp_file.write(r.content)
                        file = tmp_file.name
                else:
                    r.raise_for_status()
            except Exception as e:
                raise PDFValueError(
                    f"Errors occur in downloading the PDF file. Please check the link(s).\nError:\n{e}"
                )
        filename = os.path.splitext(os.path.basename(file))[0]

        # If the commandline has specified converting to PDF/A format
        # --compatible / -cp
        if compatible:
            with tempfile.NamedTemporaryFile(
                suffix="-pdfa.pdf", delete=False
            ) as tmp_pdfa:
                print(f"Converting {file} to PDF/A format...")
                convert_to_pdfa(file, tmp_pdfa.name)
                doc_raw = open(tmp_pdfa.name, "rb")
                os.unlink(tmp_pdfa.name)
        else:
            doc_raw = open(file, "rb")
        s_raw = doc_raw.read()
        doc_raw.close()

        temp_dir = Path(tempfile.gettempdir())
        file_path = Path(file)
        try:
            if file_path.exists() and file_path.resolve().is_relative_to(
                temp_dir.resolve()
            ):
                file_path.unlink(missing_ok=True)
                logger.debug(f"Cleaned temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean temp file {file_path}", exc_info=True)

        s_mono, s_dual = translate_stream(
            s_raw,
            **locals(),
        )
        file_mono, file_dual = get_output_paths(output, filename, lang_in, lang_out)
        if s_mono is not None:
            with open(file_mono, "wb") as doc_mono:
                doc_mono.write(s_mono)
        if s_dual is not None:
            with open(file_dual, "wb") as doc_dual:
                doc_dual.write(s_dual)
        # a file that was not requested is returned as None
        result_files.append(
            (
                str(file_mono) if s_mono is not None else None,
                str(file_dual) if s_dual is not None else None,
            )
        )

    return result_files


def download_remote_fonts(lang: str):
    lang = lang.lower()
    LANG_NAME_MAP = {
        **{la: "GoNotoKurrent-Regular.ttf" for la in noto_list},
        **{
            la: f"SourceHanSerif{region}-Regular.ttf"
            for region, langs in {
                "CN": ["zh-cn", "zh-hans", "zh"],
                "TW": ["zh-tw", "zh-hant"],
                "JP": ["ja"],
                "KR": ["ko"],
            }.items()
            for la in langs
        },
    }
    font_name = LANG_NAME_MAP.get(lang, "GoNotoKurrent-Regular.ttf")

    # docker
    font_path = ConfigManager.get("NOTO_FONT_PATH", Path("/app", font_name).as_posix())
    if not Path(font_path).exists():
        font_path, _ = get_font_and_metadata(font_name)
        font_path = font_path.as_posix()

    logger.info(f"use font: {font_path}")

    return font_path