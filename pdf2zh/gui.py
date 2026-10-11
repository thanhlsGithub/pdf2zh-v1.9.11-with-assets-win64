import asyncio
import cgi
import functools
import inspect
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from asyncio import CancelledError
from pathlib import Path
import typing as T

import gradio as gr
import requests
import tqdm
from gradio_pdf import PDF
from string import Template
import logging

from pdf2zh import __version__
from pdf2zh.high_level import translate, get_output_paths
from pdf2zh.doclayout import ModelInstance
from pdf2zh.config import ConfigManager
from pdf2zh.logsetup import setup_logging, report_exception, log_hint, get_crash_path, fmt_pages
from pdf2zh.translator import (
    AnythingLLMTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DifyTranslator,
    ArgosTranslator,
    GeminiTranslator,
    GoogleTranslator,
    GoogleNewTranslator,
    GoogleOldTranslator,
    GoogleHtmlTranslator,
    EdgeTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAITranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    GrokTranslator,
    GroqTranslator,
    DeepseekTranslator,
    OpenAIlikedTranslator,
    QwenMtTranslator,
)
from babeldoc.docvision.doclayout import OnnxModel
from babeldoc import __version__ as babeldoc_version

setup_logging("gui")  # log ra file; tiến trình dịch con import lại module này nên cũng tự cài
logger = logging.getLogger(__name__)


def _sweep_old_gradio_cache(max_age: float = 86400.0) -> None:
    """Gradio chép mỗi file tải lên vào %TEMP%/gradio và không dọn khi tắt app (mỗi sách vài chục MB).
    Mở app là xoá các mục cũ hơn `max_age` giây. Chỉ tiến trình chính; bỏ qua nếu GRADIO_TEMP_DIR
    được đặt riêng (thư mục đó do người dùng quản lý)."""
    if mp.current_process().name != "MainProcess" or os.environ.get("GRADIO_TEMP_DIR"):
        return
    root = Path(tempfile.gettempdir()) / "gradio"
    now, removed = time.time(), 0
    try:
        for entry in root.iterdir():
            try:
                if now - entry.stat().st_mtime > max_age:
                    shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
                    removed += 1
            except OSError:
                continue
    except OSError:
        return
    if removed:
        logger.info("removed %d old Gradio cache entries from %s", removed, root)


_sweep_old_gradio_cache()

_BABELDOC_MODEL = None


def _get_babeldoc_model():
    """Load the BabelDOC layout model lazily (only the worker process needs it)."""
    global _BABELDOC_MODEL
    if _BABELDOC_MODEL is None:
        _BABELDOC_MODEL = OnnxModel.load_available()
    return _BABELDOC_MODEL


# Running translation jobs: session_id -> {"proc", "stop_evt", "mode", "deadline"}
job_map: dict = {}
# "Stop" asks the worker to stop politely; if it has not exited after this many
# seconds it is killed (the translation cache is kept either way).
STOP_GRACE_SECONDS = 20

# The following variables associate strings with translators
service_map: dict[str, BaseTranslator] = {
    "Google": GoogleTranslator,
    "Google (New)": GoogleNewTranslator,
    "Google (Old)": GoogleOldTranslator,
    "Google (HTML)": GoogleHtmlTranslator,
    "Bing": BingTranslator,
    "Microsoft Edge": EdgeTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
    "Argos Translate": ArgosTranslator,
    "Grok": GrokTranslator,
    "Groq": GroqTranslator,
    "DeepSeek": DeepseekTranslator,
    "OpenAI-liked": OpenAIlikedTranslator,
    "Ali Qwen-Translation": QwenMtTranslator,
}

# The following variables associate strings with specific languages.
# The first 11 are the original ones; the rest are sorted alphabetically.
# Codes follow Google Translate. Services that use other codes (Bing, Microsoft
# Edge...) convert them in their own `lang_map` (see translator.py).
lang_map = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
    "Vietnamese": "vi",
}
lang_map.update(
    dict(
        sorted(
            {
                "Afrikaans": "af",
                "Albanian": "sq",
                "Amharic": "am",
                "Arabic": "ar",
                "Armenian": "hy",
                "Azerbaijani": "az",
                "Basque": "eu",
                "Belarusian": "be",
                "Bengali": "bn",
                "Bosnian": "bs",
                "Bulgarian": "bg",
                "Burmese": "my",
                "Catalan": "ca",
                "Cebuano": "ceb",
                "Chichewa": "ny",
                "Corsican": "co",
                "Croatian": "hr",
                "Czech": "cs",
                "Danish": "da",
                "Dutch": "nl",
                "Esperanto": "eo",
                "Estonian": "et",
                "Filipino": "tl",
                "Finnish": "fi",
                "Frisian": "fy",
                "Galician": "gl",
                "Georgian": "ka",
                "Greek": "el",
                "Gujarati": "gu",
                "Haitian Creole": "ht",
                "Hausa": "ha",
                "Hawaiian": "haw",
                "Hebrew": "he",
                "Hindi": "hi",
                "Hmong": "hmn",
                "Hungarian": "hu",
                "Icelandic": "is",
                "Igbo": "ig",
                "Indonesian": "id",
                "Irish": "ga",
                "Javanese": "jv",
                "Kannada": "kn",
                "Kazakh": "kk",
                "Khmer": "km",
                "Kinyarwanda": "rw",
                "Kurdish": "ku",
                "Kyrgyz": "ky",
                "Lao": "lo",
                "Latin": "la",
                "Latvian": "lv",
                "Lithuanian": "lt",
                "Luxembourgish": "lb",
                "Macedonian": "mk",
                "Malagasy": "mg",
                "Malay": "ms",
                "Malayalam": "ml",
                "Maltese": "mt",
                "Maori": "mi",
                "Marathi": "mr",
                "Mongolian": "mn",
                "Nepali": "ne",
                "Norwegian": "no",
                "Odia": "or",
                "Pashto": "ps",
                "Persian": "fa",
                "Polish": "pl",
                "Portuguese": "pt",
                "Punjabi": "pa",
                "Romanian": "ro",
                "Samoan": "sm",
                "Scots Gaelic": "gd",
                "Serbian": "sr",
                "Sesotho": "st",
                "Shona": "sn",
                "Sindhi": "sd",
                "Sinhala": "si",
                "Slovak": "sk",
                "Slovenian": "sl",
                "Somali": "so",
                "Sundanese": "su",
                "Swahili": "sw",
                "Swedish": "sv",
                "Tajik": "tg",
                "Tamil": "ta",
                "Tatar": "tt",
                "Telugu": "te",
                "Thai": "th",
                "Turkish": "tr",
                "Turkmen": "tk",
                "Ukrainian": "uk",
                "Urdu": "ur",
                "Uyghur": "ug",
                "Uzbek": "uz",
                "Welsh": "cy",
                "Xhosa": "xh",
                "Yiddish": "yi",
                "Yoruba": "yo",
                "Zulu": "zu",
            }.items()
        )
    )
)

# The following variable associate strings with page ranges
# "First N pages" uses one box (N); "Page range" uses two boxes (from page / to page).
page_map = {
    "All": None,
    "First": [0],
    "First N pages": None,
    "Page range": None,
}

# Check if this is a public demo, which has resource limits
flag_demo = False

# Limit resources
if ConfigManager.get("PDF2ZH_DEMO"):
    flag_demo = True
    service_map = {
        "Google": GoogleTranslator,
    }
    page_map = {
        "First": [0],
        "First 20 pages": list(range(0, 20)),
    }
    client_key = ConfigManager.get("PDF2ZH_CLIENT_KEY")
    server_key = ConfigManager.get("PDF2ZH_SERVER_KEY")


# Limit Enabled Services
enabled_services: T.Optional[T.List[str]] = ConfigManager.get("ENABLED_SERVICES")
if isinstance(enabled_services, list):
    default_services = ["Google", "Bing"]
    enabled_services_names = [str(_).lower().strip() for _ in enabled_services]
    enabled_services = [
        k
        for k in service_map.keys()
        if str(k).lower().strip() in enabled_services_names
    ]
    if len(enabled_services) == 0:
        raise RuntimeError("No services available.")
    enabled_services = default_services + enabled_services
else:
    enabled_services = list(service_map.keys())

# Default engine on the web page: "Google (New)" (falls back to the first
# available one if it is not enabled, e.g. in demo mode)
default_engine = (
    "Google (New)" if "Google (New)" in enabled_services else enabled_services[0]
)


# Configure about Gradio show keys
hidden_gradio_details: bool = bool(ConfigManager.get("HIDDEN_GRADIO_DETAILS"))


# Public demo control
def verify_recaptcha(response):
    """
    This function verifies the reCAPTCHA response.
    """
    recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
    data = {"secret": server_key, "response": response}
    result = requests.post(recaptcha_url, data=data).json()
    return result.get("success")


def download_with_limit(url: str, save_path: str, size_limit: int) -> str:
    """
    This function downloads a file from a URL and saves it to a specified path.

    Inputs:
        - url: The URL to download the file from
        - save_path: The path to save the file to
        - size_limit: The maximum size of the file to download

    Returns:
        - The path of the downloaded file
    """
    chunk_size = 1024
    total_size = 0
    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        content = response.headers.get("Content-Disposition")
        try:  # filename from header
            _, params = cgi.parse_header(content)
            filename = params["filename"]
        except Exception:  # filename from url
            filename = os.path.basename(url)
        filename = os.path.splitext(os.path.basename(filename))[0] + ".pdf"
        with open(save_path / filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                total_size += len(chunk)
                if size_limit and total_size > size_limit:
                    raise gr.Error("Exceeds file size limit")
                file.write(chunk)
    return save_path / filename


def _kill_process_tree(proc) -> None:
    """Hard-kill a worker process and everything it started.
    (A thread has no pid, so this does nothing for thread workers.)"""
    pid = getattr(proc, "pid", None)
    if pid is None:
        return
    try:
        import psutil

        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            parent.kill()
        except psutil.NoSuchProcess:
            pass
        return
    except ImportError:
        pass
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            proc.kill()
        except Exception:
            pass


def _find_job(state: dict):
    """Job of this browser session; with a single running job, use that one."""
    session_id = (state or {}).get("session_id")
    if session_id in job_map:
        return job_map[session_id]
    if len(job_map) == 1:
        return next(iter(job_map.values()))
    return None


# One dynamic button: it is "Stop" while a translation runs (or nothing runs)
# and turns into "Continue" after a stop.
LABEL_STOP = "Stop — Tạm dừng (giữ cache)"
LABEL_CONTINUE = "Continue — Tiếp tục dịch"


def _toggle_update(paused: bool):
    """Label / colour of the Stop <-> Continue button."""
    return gr.update(
        value=LABEL_CONTINUE if paused else LABEL_STOP,
        variant="primary" if paused else "secondary",
    )


def sync_toggle(state: dict):
    """Set the Stop/Continue button from the session state."""
    return _toggle_update(bool((state or {}).get("paused")))


def _no_change(n: int = 6) -> tuple:
    """`n` "leave this output alone" updates."""
    return tuple(gr.update() for _ in range(n))


def stop_translate_file(state: dict):
    """
    STOP (can continue): ask the worker to stop at the next safe point. If it
    does not react within STOP_GRACE_SECONDS it is killed. Already translated
    text stays in the cache. The button turns into "Continue"; clicking it
    translates again and reuses the cache (keep "Ignore cache" unchecked), so
    it picks up quickly where it stopped.
    """
    job = _find_job(state)
    if job is None:
        gr.Info("No translation is running")
        return sync_toggle(state)
    logger.info("Stopping translation (cache kept)")
    job["mode"] = "stop"
    job["deadline"] = time.monotonic() + STOP_GRACE_SECONDS
    job["stop_evt"].set()
    state["paused"] = True
    gr.Info("Stopping... Click Continue to resume (translated text is cached)")
    return _toggle_update(True)


def exit_translate_file(state: dict):
    """
    EXIT (kill): terminate the translation process immediately, including
    everything it started. In thread mode (PyStand) the thread cannot be
    killed: it is told to stop and the GUI stops waiting for it.
    """
    job = _find_job(state)
    if job is None:
        gr.Info("No translation is running")
        return sync_toggle(state)
    logger.info("Killing translation process")
    job["mode"] = "kill"
    job["stop_evt"].set()
    _kill_process_tree(job["proc"])
    state["paused"] = False
    gr.Info("Translation process terminated")
    return _toggle_update(False)


def _wait_previous_job(state: dict) -> None:
    """If the previous worker of this session is still shutting down (after
    Stop / Exit), wait for it so that two workers never run at the same time."""
    old = (state or {}).get("session_id")
    deadline = time.monotonic() + STOP_GRACE_SECONDS + 10
    while time.monotonic() < deadline:
        job = job_map.get(old)
        if job is None or job["mode"] is None:
            return
        time.sleep(0.3)


def _translation_worker(use_babeldoc, param, stop_evt, msg_q) -> None:
    """Runs in a separate process (or a thread): does the actual translation."""
    try:
        setup_logging("worker")
        logger.info(
            "worker start: babeldoc=%s files=%s pages=%s %s->%s service=%s thread=%s format=%s",
            use_babeldoc, param.get("files"), fmt_pages(param.get("pages")),
            param.get("lang_in"), param.get("lang_out"), param.get("service"),
            param.get("thread"), param.get("output_format"),
        )
        from pdf2zh import doclayout as _dl
        from pdf2zh import translator as _tr

        # every translator created from now on stops as soon as "Stop" is
        # pressed (works in process mode and in thread mode)
        _tr.set_cancel_event(stop_evt)

        if ModelInstance.value is None:
            ModelInstance.value = getattr(_dl, "OnnxModel", OnnxModel).load_available()

        last = [0.0]

        def _callback(t):
            now = time.monotonic()
            if now - last[0] < 0.2:
                return
            last[0] = now
            try:
                total = getattr(t, "total", None) or 0
                frac = (t.n / total) if total else 0.0
                desc = getattr(t, "desc", "") or "Translating..."
                msg_q.put(("progress", min(max(frac, 0.0), 1.0), desc))
            except Exception:
                pass

        param = dict(param)
        param["callback"] = _callback
        param["cancellation_event"] = stop_evt
        param["model"] = ModelInstance.value

        if use_babeldoc:
            res = babeldoc_translate_file(**param)
            # build_outputs() tuple -> (mono, preview, dual, ...)
            msg_q.put(("done", (res[0], res[2])))
        else:
            translate(**param)
            msg_q.put(("done", None))
    except CancelledError:
        logger.info("worker: translation cancelled")
        msg_q.put(("cancelled", None))
    except BaseException as e:  # noqa: BLE001 - report everything to the parent
        # traceback đầy đủ vào pdf2zh_logs/pdf2zh.log; GUI nhận bản tóm tắt + đường dẫn log
        msg_q.put(("error", report_exception("translation worker", e)))


def _can_spawn_process() -> bool:
    """spawn needs a real python.exe. PyStand / frozen builds re-run the whole
    app for the child process (-> "unrecognized arguments: -S -I
    --multiprocessing-fork", exit code 2), so they fall back to a thread.
    Override with the environment variable PDF2ZH_USE_PROCESS=1 (force process)
    or PDF2ZH_USE_PROCESS=0 (force thread)."""
    flag = str(os.environ.get("PDF2ZH_USE_PROCESS", "")).strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    if getattr(sys, "frozen", False):
        return False
    return os.path.basename(sys.executable or "").lower().startswith("python")


def run_translation_job(session_id, use_babeldoc, param, progress):
    """
    Run the translation in a separate process (real python, so that it can
    really be stopped) or, when spawn is not possible (PyStand), in a thread.
    Returns the payload of the worker ("done" message).
    """
    use_process = _can_spawn_process()
    if use_process:
        ctx = mp.get_context("spawn")
        stop_evt = ctx.Event()
        msg_q = ctx.Queue()
        proc = ctx.Process(
            target=_translation_worker,
            args=(use_babeldoc, param, stop_evt, msg_q),
            daemon=True,
        )
    else:
        logger.info("spawn not available here, running the translation in a thread")
        print("[pdf2zh-fix] spawn not available here, running the translation in a thread")
        stop_evt = threading.Event()
        msg_q = queue.Queue()
        proc = threading.Thread(
            target=_translation_worker,
            args=(use_babeldoc, param, stop_evt, msg_q),
            daemon=True,
        )
    job = {"proc": proc, "stop_evt": stop_evt, "mode": None, "deadline": None}
    job_map[session_id] = job
    proc.start()

    result = None
    try:
        while True:
            try:
                msg = msg_q.get(timeout=0.5)
            except queue.Empty:
                msg = None
            if msg is not None:
                if msg[0] == "progress":
                    progress(msg[1], desc=msg[2])
                    continue
                result = msg
                break
            if not use_process and (
                job["mode"] == "kill"
                or (
                    job["mode"] == "stop"
                    and job["deadline"] is not None
                    and time.monotonic() > job["deadline"]
                    and proc.is_alive()
                )
            ):
                # a thread cannot be killed: stop waiting for it (it ends by
                # itself at its next translate call because stop_evt is set)
                break
            if (
                use_process
                and job["mode"] == "stop"
                and job["deadline"] is not None
                and time.monotonic() > job["deadline"]
                and proc.is_alive()
            ):
                logger.info("Worker did not stop in time, killing it")
                _kill_process_tree(proc)
            if not proc.is_alive():
                try:  # a final message may still be in the queue
                    result = msg_q.get(timeout=1)
                except queue.Empty:
                    result = None
                break
    finally:
        if use_process and proc.is_alive():
            _kill_process_tree(proc)
        # do not wait for a thread that is being abandoned after Stop / Exit
        proc.join(timeout=5 if (use_process or job["mode"] not in ("kill", "stop")) else 0)
        job_map.pop(session_id, None)
        if use_process:
            try:
                msg_q.close()
                msg_q.cancel_join_thread()
            except Exception:
                pass

    if result is not None and result[0] == "done":
        return result[1]
    if job["mode"] == "kill":
        raise gr.Error("Translation process terminated")
    if job["mode"] == "stop" or (result is not None and result[0] == "cancelled"):
        raise gr.Error(
            "Translation stopped. Click Translate to continue "
            "(already translated text is cached)"
        )
    if result is not None and result[0] == "error":
        raise gr.Error(result[1])
    crash = get_crash_path()
    msg = (
        "Translation process ended unexpectedly "
        f"(exit code {getattr(proc, 'exitcode', None)}). "
        f"{log_hint()}" + (f"\nCrash dump: {crash}" if crash else "")
    )
    logger.error(msg)
    raise gr.Error(msg)


def get_page_count(pdf_path) -> T.Optional[int]:
    """Number of pages of a PDF, or None if it cannot be read."""
    try:
        from pymupdf import Document

        with Document(str(pdf_path)) as doc:
            return doc.page_count
    except Exception:
        return None


def build_outputs(file_mono, file_dual, output_format: str = "mono"):
    """
    Build the 6 values expected by the Gradio outputs, honouring the chosen
    output format ("mono", "dual" or "both"). The download box of the file that
    was not asked for is hidden.
    """
    want_mono = output_format in ("mono", "both")
    want_dual = output_format in ("dual", "both")

    mono_out = str(file_mono) if (want_mono and file_mono) else None
    dual_out = str(file_dual) if (want_dual and file_dual) else None
    preview = mono_out or dual_out
    return (
        mono_out,
        preview,
        dual_out,
        gr.update(visible=want_mono),
        gr.update(visible=want_dual),
        gr.update(visible=True),
    )


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _discard_input(path) -> None:
    """Xoá bản sao file gốc trong pdf2zh_files sau khi dịch (đã có file gốc của bạn + bản dịch)."""
    if path is None:
        return
    try:
        Path(path).unlink()
        logger.debug("removed input copy %s", path)
    except FileNotFoundError:
        pass
    except OSError as e:  # đang bị khoá: để lại, lần sau tự dọn không cần thiết
        logger.debug("could not remove input copy %s: %s", path, e)


def _logged(fn):
    """Gradio chỉ hiện chữ "Error" cho exception thường. Bọc lại: ghi traceback vào log và
    đổi thành gr.Error có nội dung + đường dẫn log."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gr.Error as e:
            logger.warning("%s -> %s", fn.__name__, e)
            raise
        except CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            raise gr.Error(report_exception(fn.__name__, e)) from e

    return wrapper


@_logged
def translate_file(
    file_type,
    file_input,
    link_input,
    service,
    lang_from,
    lang_to,
    page_range,
    page_input,
    page_from,
    page_to,
    output_format,
    prompt,
    threads,
    skip_subset_fonts,
    ignore_cache,
    vfont,
    use_babeldoc,
    recaptcha_response,
    state,
    progress=gr.Progress(),
    *envs,
):
    """
    This function translates a PDF file from one language to another.

    Inputs:
        - file_type: The type of file to translate
        - file_input: The file to translate
        - link_input: The link to the file to translate
        - service: The translation service (engine) to use
        - lang_from: The language to translate from
        - lang_to: The language to translate to
        - page_range: The range of pages to translate
        - page_input: The input for "First N pages"
        - page_from: First page to translate (for "Page range")
        - page_to: Last page to translate (for "Page range")
        - output_format: "mono", "dual" or "both"
        - prompt: The custom prompt for the llm
        - threads: The number of threads to use
        - recaptcha_response: The reCAPTCHA response
        - state: The state of the translation process
        - progress: The progress bar
        - envs: The environment variables

    Returns:
        - The translated file
        - The translated file
        - The translated file
        - The progress bar
        - The progress bar
        - The progress bar
    """
    _wait_previous_job(state)
    session_id = uuid.uuid4()
    state["session_id"] = session_id
    state["paused"] = False
    # Translate PDF content using selected service.
    if flag_demo and not verify_recaptcha(recaptcha_response):
        raise gr.Error("reCAPTCHA fail")

    progress(0, desc="Starting translation...")

    output = Path("pdf2zh_files")
    output.mkdir(parents=True, exist_ok=True)
    before = set(os.listdir(output))  # để biết bản sao đầu vào nào do chính lần dịch này tạo ra

    if file_type == "File":
        if not file_input:
            raise gr.Error("No input")
        file_path = shutil.copy(file_input, output)
    else:
        if not link_input:
            raise gr.Error("No input")
        file_path = download_with_limit(
            link_input,
            output,
            5 * 1024 * 1024 if flag_demo else None,
        )

    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_raw = output / f"{filename}.pdf"
    # Bản sao đầu vào chỉ là phần thừa khi dịch xong. Chỉ xoá nếu nó do lần này tạo ra (file đã có sẵn
    # trong pdf2zh_files từ trước thì không đụng tới). Đặt PDF2ZH_KEEP_INPUT=1 để giữ lại.
    discard = None if (file_raw.name in before or _env_flag("PDF2ZH_KEEP_INPUT")) else file_raw

    try:
        if output_format not in ("mono", "dual", "both"):
            output_format = "mono"

        translator = service_map[service]
        if page_range == "First N pages":
            try:
                n_pages = int(str(page_input).strip())
                if n_pages < 1:
                    raise ValueError
            except ValueError:
                raise gr.Error("Please enter a valid number of pages (e.g. 20)")
            selected_page = list(range(0, n_pages))
        elif page_range == "Page range":
            try:
                p_from = int(str(page_from).strip())
                p_to = int(str(page_to).strip())
            except ValueError:
                raise gr.Error(
                    "Please enter both 'From page' and 'To page' (e.g. 50 and 100)"
                )
            if p_from < 1 or p_to < p_from:
                raise gr.Error(
                    "Invalid page range: 'From page' must be at least 1 and not greater than 'To page'"
                )
            total_pages = get_page_count(file_raw)
            if total_pages is not None:
                if p_from > total_pages:
                    raise gr.Error(f"The document only has {total_pages} pages")
                p_to = min(p_to, total_pages)  # 'To page' beyond the end -> last page
            selected_page = list(range(p_from - 1, p_to))
        else:
            selected_page = page_map[page_range]
        lang_from = lang_map[lang_from]
        lang_to = lang_map[lang_to]
        # e.g. <name>-VI.pdf (mono) and <name>-EN-VI.pdf (dual)
        file_mono, file_dual = get_output_paths(output, filename, lang_from, lang_to)
        need_mono = output_format in ("mono", "both")
        need_dual = output_format in ("dual", "both")

        _envs = {}
        for i, env in enumerate(translator.envs.items()):
            _envs[env[0]] = envs[i]
        for k, v in _envs.items():
            if str(k).upper().endswith("API_KEY") and str(v) == "***":
                # Load Real API_KEYs from local configure file
                real_keys: str = ConfigManager.get_env_by_translatername(
                    translator, k, None
                )
                _envs[k] = real_keys

        try:
            threads = int(threads)
        except ValueError:
            threads = 1

        param = {
            "files": [str(file_raw)],
            "pages": selected_page,
            "lang_in": lang_from,
            "lang_out": lang_to,
            "service": f"{translator.name}",
            "output": output,
            "thread": int(threads),
            "envs": _envs,
            "prompt": Template(prompt) if prompt else None,
            "skip_subset_fonts": skip_subset_fonts,
            "ignore_cache": ignore_cache,
            "vfont": vfont,  # 添加自定义公式字体正则表达式
            "output_format": output_format,
        }

        # Runs in a separate process or thread (callback / cancellation event /
        # model are added inside the worker). Raises gr.Error when stopped/killed.
        result = run_translation_job(session_id, use_babeldoc, param, progress)
        state["paused"] = False  # finished normally (even if Stop was clicked too late)
        if use_babeldoc:
            progress(1.0, desc="Translation complete!")
            return build_outputs(result[0], result[1], output_format)

        if (need_mono and not file_mono.exists()) or (
            need_dual and not file_dual.exists()
        ):
            raise gr.Error("No output")

        progress(1.0, desc="Translation complete!")

        return build_outputs(
            file_mono if need_mono else None,
            file_dual if need_dual else None,
            output_format,
        )
    finally:
        _discard_input(discard)


_toggle_lock = threading.Lock()


def toggle_translate(
    btn_label,
    file_type,
    file_input,
    link_input,
    service,
    lang_from,
    lang_to,
    page_range,
    page_input,
    page_from,
    page_to,
    output_format,
    prompt,
    threads,
    skip_subset_fonts,
    ignore_cache,
    vfont,
    use_babeldoc,
    recaptcha_response,
    state,
    progress=gr.Progress(),
    *envs,
):
    """
    ONE button that changes its name and its behaviour:
      - label "Stop"     -> stop the translation; the label becomes "Continue"
      - label "Continue" -> translate again (the cache makes it pick up quickly);
                            the label becomes "Stop"
    The label the browser sent decides the action, so a quick double click on
    "Stop" can never be taken for a "Continue".
    Outputs: the 6 outputs of translate_file + the button itself.
    """
    if btn_label != LABEL_CONTINUE:
        yield (*_no_change(), stop_translate_file(state))
        return

    with _toggle_lock:
        job = job_map.get(state.get("session_id"))
        if state.get("busy") or (job is not None and job["mode"] is None):
            gr.Info("A translation is already running")
            yield (*_no_change(), _toggle_update(False))
            return
        state["busy"] = True
    try:
        yield (*_no_change(), _toggle_update(False))  # show "Stop" straight away
        out = translate_file(
            file_type,
            file_input,
            link_input,
            service,
            lang_from,
            lang_to,
            page_range,
            page_input,
            page_from,
            page_to,
            output_format,
            prompt,
            threads,
            skip_subset_fonts,
            ignore_cache,
            vfont,
            use_babeldoc,
            recaptcha_response,
            state,
            progress,
            *envs,
        )
        yield (*out, _toggle_update(False))
    finally:
        state["busy"] = False


# ---------------------------------------------------------------------------
# BabelDOC helpers
#
# - BabelDOC reports "finish" BEFORE its output PDF is completely written
#   (the old rename code got FileNotFoundError), so we wait for the file.
# - BabelDOC's own "dual" PDF puts original + translation side by side on ONE
#   page. pdf2zh's classic engine puts them on alternating pages (original,
#   translated, original, translated...). We reproduce the classic layout.
# ---------------------------------------------------------------------------
def _cfg_accepts(cfg_cls, name: str) -> bool:
    """True if the installed BabelDOC TranslationConfig has a parameter `name`."""
    try:
        return name in inspect.signature(cfg_cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _wait_for_file(path, wait: float = 180.0, delay: float = 0.5) -> bool:
    """Wait until `path` exists and is not empty."""
    deadline = time.monotonic() + wait
    p = Path(path)
    while time.monotonic() < deadline:
        try:
            if p.exists() and p.stat().st_size > 0:
                return True
        except OSError:
            pass
        time.sleep(delay)
    return False


def _move_with_retry(src, dst, wait: float = 180.0, delay: float = 0.5) -> Path:
    """Rename src -> dst, waiting for `src` to appear / be unlocked (Windows).

    Falls back to copy, and finally returns `src` (keeps BabelDOC's own name)
    while PRINTING the error instead of swallowing it silently.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        return dst
    last = None
    deadline = time.monotonic() + wait
    while True:
        try:
            if src.exists():
                dst.unlink(missing_ok=True)
                src.replace(dst)
                return dst
        except OSError as e:  # PermissionError: the file is still locked
            last = e
        if time.monotonic() >= deadline:
            break
        time.sleep(delay)
    try:
        shutil.copyfile(src, dst)
        try:
            src.unlink()
        except OSError:
            pass
        return dst
    except OSError as e:
        msg = f"Cannot rename {src} -> {dst}: {last!r} / {e!r}"
        print(f"[pdf2zh-fix] {msg}")
        logger.error(msg)
        return src


def _find_babeldoc_output(out_dir, stem: str, kind: str):
    """Find BabelDOC's own output <stem>.<lang>.<mono|dual>.pdf (newest one)."""
    best = None
    for p in Path(out_dir).iterdir():
        n = p.name
        if n.startswith(stem + ".") and n.endswith(f".{kind}.pdf"):
            if best is None or p.stat().st_mtime > best.stat().st_mtime:
                best = p
    return best


def _make_alternating_dual(original_pdf, mono_pdf, out_pdf, selected_pages=None) -> bool:
    """Build the classic pdf2zh bilingual PDF: original page, translated page,
    original page, translated page... (same algorithm as insert_file +
    move_page in high_level.translate_stream)."""
    from pymupdf import Document

    doc_en = Document(str(original_pdf))
    doc_zh = Document(str(mono_pdf))
    try:
        n, m = doc_en.page_count, doc_zh.page_count
        if m == n:
            doc_en.insert_file(doc_zh)
            for i in range(n):
                doc_en.move_page(n + i, i * 2 + 1)
            doc_out = doc_en
        elif selected_pages and m == len(selected_pages):
            # BabelDOC only exported the selected pages -> pair them up
            doc_out = Document()
            for k, pno in enumerate(sorted(selected_pages)):
                doc_out.insert_pdf(doc_en, from_page=pno, to_page=pno)
                doc_out.insert_pdf(doc_zh, from_page=k, to_page=k)
        else:
            logger.warning(
                f"Cannot pair pages: original={n}, translated={m}, "
                f"selected={len(selected_pages or [])}"
            )
            return False
        try:
            doc_out.subset_fonts(fallback=True)
        except Exception:
            pass
        doc_out.save(str(out_pdf), deflate=True, garbage=3, use_objstms=1)
        return True
    finally:
        doc_zh.close()
        doc_en.close()


def babeldoc_translate_file(**kwargs):
    from babeldoc.high_level import init as babeldoc_init

    babeldoc_init()
    from babeldoc.high_level import async_translate as babeldoc_translate
    from babeldoc.translation_config import TranslationConfig as YadtConfig

    output_format = kwargs.get("output_format", "mono")

    for translator in [
        GoogleTranslator,
        GoogleNewTranslator,
        GoogleOldTranslator,
        GoogleHtmlTranslator,
        BingTranslator,
        EdgeTranslator,
        DeepLTranslator,
        DeepLXTranslator,
        OllamaTranslator,
        XinferenceTranslator,
        AzureOpenAITranslator,
        OpenAITranslator,
        ZhipuTranslator,
        ModelScopeTranslator,
        SiliconTranslator,
        GeminiTranslator,
        AzureTranslator,
        TencentTranslator,
        DifyTranslator,
        AnythingLLMTranslator,
        ArgosTranslator,
        GrokTranslator,
        GroqTranslator,
        DeepseekTranslator,
        OpenAIlikedTranslator,
        QwenMtTranslator,
    ]:
        if kwargs["service"] == translator.name:
            translator = translator(
                kwargs["lang_in"],
                kwargs["lang_out"],
                "",
                envs=kwargs["envs"],
                prompt=kwargs["prompt"],
                ignore_cache=kwargs["ignore_cache"],
            )
            break
    else:
        raise ValueError("Unsupported translation service")
    import asyncio
    from babeldoc.main import create_progress_handler

    want_mono = output_format in ("mono", "both")
    want_dual = output_format in ("dual", "both")

    # Newer BabelDOC has a native "alternating pages" dual mode. Older ones do
    # not -> build the alternating bilingual PDF ourselves from the mono file
    # and the original (same result as the classic engine).
    native_alt = _cfg_accepts(YadtConfig, "use_alternating_pages_dual")
    need_dual = want_dual and native_alt
    need_mono = want_mono or (want_dual and not native_alt)

    # Optional (off by default): make BabelDOC split short lines into separate
    # paragraphs (helps tables of contents). Enable with the environment
    # variable / config key BABELDOC_SPLIT_SHORT_LINES=1
    split_lines = str(
        ConfigManager.get("BABELDOC_SPLIT_SHORT_LINES", "") or ""
    ).strip().lower() in ("1", "true", "yes", "on")

    for file in kwargs["files"]:
        file = file.strip("\"'")
        filename = os.path.splitext(os.path.basename(file))[0]

        # `pages` is a list of 0-based page numbers (None = all pages);
        # BabelDOC expects a 1-based, comma-separated string.
        selected_pages = kwargs.get("pages") or []
        pages_str = ",".join(str(p + 1) for p in selected_pages)

        cfg = dict(
            input_file=file,
            font=None,
            pages=pages_str,
            output_dir=kwargs["output"],
            doc_layout_model=_get_babeldoc_model(),
            translator=translator,
            debug=False,
            lang_in=kwargs["lang_in"],
            lang_out=kwargs["lang_out"],
            no_dual=not need_dual,
            no_mono=not need_mono,
            qps=kwargs["thread"],
            use_rich_pbar=False,
            disable_rich_text_translate=not isinstance(translator, OpenAITranslator),
            skip_clean=kwargs["skip_subset_fonts"],
            report_interval=0.5,
        )
        if native_alt:
            cfg["use_alternating_pages_dual"] = True
        if split_lines:
            if _cfg_accepts(YadtConfig, "split_short_lines"):
                cfg["split_short_lines"] = True
            else:
                print("[pdf2zh-fix] this BabelDOC has no split_short_lines option")
        yadt_config = YadtConfig(**cfg)

        async def yadt_translate_coro(yadt_config):
            progress_context, progress_handler = create_progress_handler(yadt_config)
            result = None
            agen = babeldoc_translate(yadt_config)
            try:
                with progress_context:
                    async for event in agen:
                        progress_handler(event)
                        if yadt_config.debug:
                            logger.debug(event)
                        kwargs["callback"](progress_context)
                        if kwargs["cancellation_event"].is_set():
                            yadt_config.cancel_translation()
                            raise CancelledError
                        if event["type"] == "finish":
                            result = event["translate_result"]
                            logger.info("Translation Result:")
                            logger.info(f"  Original PDF: {result.original_pdf_path}")
                            logger.info(f"  Time Cost: {result.total_seconds:.2f}s")
                            logger.info(f"  Mono PDF: {result.mono_pdf_path or 'None'}")
                            logger.info(f"  Dual PDF: {result.dual_pdf_path or 'None'}")
                            break
            finally:
                # `break` leaves BabelDOC's generator suspended while its worker
                # is still finishing. Close it so the worker can complete.
                aclose = getattr(agen, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        logger.debug("aclose failed", exc_info=True)
            import gc

            gc.collect()

            out_dir = Path(kwargs["output"])
            file_mono = (
                Path(result.mono_pdf_path) if result and result.mono_pdf_path else None
            )
            file_dual = (
                Path(result.dual_pdf_path) if result and result.dual_pdf_path else None
            )
            # in case this BabelDOC version does not report the paths
            if need_mono and file_mono is None:
                file_mono = _find_babeldoc_output(out_dir, filename, "mono")
            if need_dual and file_dual is None:
                file_dual = _find_babeldoc_output(out_dir, filename, "dual")
            # "finish" can arrive before the files are completely written -> wait
            for _f in (file_mono, file_dual):
                if _f is not None and not _wait_for_file(_f):
                    print(f"[pdf2zh-fix] BabelDOC output not found: {_f}")

            # <name>-VI-BabelDOC.pdf  and  <name>-EN-VI-BabelDOC.pdf
            target_mono, target_dual = get_output_paths(
                out_dir,
                filename,
                kwargs["lang_in"],
                kwargs["lang_out"],
                suffix="-BabelDOC",
            )

            if want_dual and not native_alt:
                # original page, translated page, original page, ...
                if file_mono and _make_alternating_dual(
                    file, file_mono, target_dual, selected_pages
                ):
                    file_dual = target_dual
                else:
                    file_dual = None
                    gr.Warning("Could not build the alternating dual PDF")

            if want_mono and file_mono:
                file_mono = _move_with_retry(file_mono, target_mono)
            elif file_mono:  # only produced to build the dual PDF -> remove it
                try:
                    file_mono.unlink()
                except OSError:
                    pass
                file_mono = None

            if file_dual and native_alt:
                file_dual = _move_with_retry(file_dual, target_dual)

            print(f"[pdf2zh-fix] BabelDOC output: mono={file_mono} dual={file_dual}")
            if not (file_mono or file_dual):
                raise gr.Error("No output")
            return build_outputs(
                file_mono if want_mono else None,
                file_dual if want_dual else None,
                output_format,
            )

        return asyncio.run(yadt_translate_coro(yadt_config))


# Global setup
custom_blue = gr.themes.Color(
    c50="#E8F3FF",
    c100="#BEDAFF",
    c200="#94BFFF",
    c300="#6AA1FF",
    c400="#4080FF",
    c500="#165DFF",  # Primary color
    c600="#0E42D2",
    c700="#0A2BA6",
    c800="#061D79",
    c900="#03114D",
    c950="#020B33",
)

custom_css = """
    .secondary-text {color: #999 !important;}
    footer {visibility: hidden}
    .env-warning {color: #dd5500 !important;}
    .env-success {color: #559900 !important;}

    /* Add dashed border to input-file class */
    .input-file {
        border: 1.2px dashed #165DFF !important;
        border-radius: 6px !important;
    }

    .progress-bar-wrap {
        border-radius: 8px !important;
    }

    .progress-bar {
        border-radius: 8px !important;
    }

    .pdf-canvas canvas {
        width: 100%;
    }
    """

demo_recaptcha = """
    <script src="https://www.google.com/recaptcha/api.js?render=explicit" async defer></script>
    <script type="text/javascript">
        var onVerify = function(token) {
            el=document.getElementById('verify').getElementsByTagName('textarea')[0];
            el.value=token;
            el.dispatchEvent(new Event('input'));
        };
    </script>
    """

tech_details_string = f"""
                    <summary>Technical details</summary>
                    - GitHub: <a href="https://github.com/Byaidu/PDFMathTranslate">Byaidu/PDFMathTranslate</a><br>
                    - BabelDOC: <a href="https://github.com/funstory-ai/BabelDOC">funstory-ai/BabelDOC</a><br>
                    - GUI by: <a href="https://github.com/reycn">Rongxin</a><br>
                    - pdf2zh Version: {__version__} <br>
                    - BabelDOC Version: {babeldoc_version}
                """


# Gradio tự xoá các file tải lên cũ hơn 1 ngày (kiểm tra mỗi giờ) nếu bản Gradio hỗ trợ
_blocks_extra = {}
if "delete_cache" in inspect.signature(gr.Blocks.__init__).parameters:
    _blocks_extra["delete_cache"] = (3600, 86400)

# The following code creates the GUI
with gr.Blocks(
    title="PDFMathTranslate - PDF Translation with preserved formats",
    theme=gr.themes.Default(
        primary_hue=custom_blue, spacing_size="md", radius_size="lg"
    ),
    css=custom_css,
    head=demo_recaptcha if flag_demo else "",
    **_blocks_extra,
) as demo:
    gr.Markdown(
        "# [PDFMathTranslate @ GitHub](https://github.com/Byaidu/PDFMathTranslate)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## File | < 5 MB" if flag_demo else "## File")
            file_type = gr.Radio(
                choices=["File", "Link"],
                label="Type",
                value="File",
            )
            file_input = gr.File(
                label="File",
                file_count="single",
                file_types=[".pdf"],
                type="filepath",
                elem_classes=["input-file"],
            )
            link_input = gr.Textbox(
                label="Link",
                visible=False,
                interactive=True,
            )
            gr.Markdown("## Option")
            service = gr.Dropdown(
                label="Engine",
                choices=enabled_services,
                value=default_engine,
            )
            envs = []
            for i in range(3):
                envs.append(
                    gr.Textbox(
                        visible=False,
                        interactive=True,
                    )
                )
            with gr.Row():
                lang_from = gr.Dropdown(
                    label="Translate from",
                    choices=lang_map.keys(),
                    value=ConfigManager.get("PDF2ZH_LANG_FROM", "English"),
                )
                lang_to = gr.Dropdown(
                    label="Translate to",
                    choices=lang_map.keys(),
                    value=ConfigManager.get("PDF2ZH_LANG_TO", "Vietnamese"),
                )
            page_range = gr.Radio(
                choices=page_map.keys(),
                label="Pages",
                value=list(page_map.keys())[0],
            )

            # "First N pages": a single box
            page_input = gr.Textbox(
                label="Number of pages",
                placeholder="e.g. 20",
                visible=False,
                interactive=True,
            )
            # "Page range": translate from page N to page M (e.g. 50 and 100 -> pages 50-100)
            with gr.Row(visible=False) as page_range_row:
                page_from = gr.Textbox(
                    label="From page",
                    placeholder="e.g. 50",
                    interactive=True,
                )
                page_to = gr.Textbox(
                    label="To page",
                    placeholder="e.g. 100",
                    interactive=True,
                )

            output_format = gr.Radio(
                choices=[
                    ("Mono only — Chỉ xuất PDF đã dịch", "mono"),
                    ("Dual only — Chỉ xuất PDF song ngữ", "dual"),
                    ("Both — Xuất cả hai bản", "both"),
                ],
                label="Output format — Định dạng đầu ra",
                value="mono",
            )

            with gr.Accordion("Open for More Experimental Options!", open=False):
                gr.Markdown("#### Experimental")
                threads = gr.Textbox(
                    label="number of threads", interactive=True, value="4"
                )
                skip_subset_fonts = gr.Checkbox(
                    label="Skip font subsetting", interactive=True, value=False
                )
                ignore_cache = gr.Checkbox(
                    label="Ignore cache", interactive=True, value=False
                )
                vfont = gr.Textbox(
                    label="Custom formula font regex (vfont)",
                    interactive=True,
                    value=ConfigManager.get("PDF2ZH_VFONT", ""),
                )
                prompt = gr.Textbox(
                    label="Custom Prompt for llm", interactive=True, visible=False
                )
                use_babeldoc = gr.Checkbox(
                    label="Use BabelDOC", interactive=True, value=False
                )
                envs.append(prompt)

            def on_select_service(service, evt: gr.EventData):
                translator = service_map[service]
                _envs = []
                for i in range(4):
                    _envs.append(gr.update(visible=False, value=""))
                for i, env in enumerate(translator.envs.items()):
                    label = env[0]
                    value = ConfigManager.get_env_by_translatername(
                        translator, env[0], env[1]
                    )
                    visible = True
                    if hidden_gradio_details:
                        if (
                            "MODEL" not in str(label).upper()
                            and value
                            and hidden_gradio_details
                        ):
                            visible = False
                        # Hidden Keys From Gradio
                        if "API_KEY" in label.upper():
                            value = "***"  # We use "***" Present Real API_KEY
                    _envs[i] = gr.update(
                        visible=visible,
                        label=label,
                        value=value,
                    )
                _envs[-1] = gr.update(visible=translator.CustomPrompt)
                return _envs

            def on_select_filetype(file_type):
                return (
                    gr.update(visible=file_type == "File"),
                    gr.update(visible=file_type == "Link"),
                )

            def on_select_page(choice):
                # outputs: [page_input (First N pages), page_range_row (From/To page)]
                if choice == "First N pages":
                    return (
                        gr.update(
                            visible=True,
                            label="Number of pages",
                            placeholder="e.g. 20",
                            value="",
                        ),
                        gr.update(visible=False),
                    )
                elif choice == "Page range":
                    return (
                        gr.update(visible=False),
                        gr.update(visible=True),
                    )
                else:
                    return gr.update(visible=False), gr.update(visible=False)

            def on_vfont_change(value):
                ConfigManager.set("PDF2ZH_VFONT", value)
                return value

            output_title = gr.Markdown("## Translated", visible=False)
            output_file_mono = gr.File(
                label="Download Translation (Mono)", visible=False
            )
            output_file_dual = gr.File(
                label="Download Translation (Dual)", visible=False
            )
            recaptcha_response = gr.Textbox(
                label="reCAPTCHA Response", elem_id="verify", visible=False
            )
            recaptcha_box = gr.HTML('<div id="recaptcha-box"></div>')
            translate_btn = gr.Button("Translate", variant="primary")
            # One dynamic button: "Stop" -> (after a stop) "Continue" -> "Stop" ...
            toggle_btn = gr.Button(LABEL_STOP, variant="secondary")
            cancellation_btn = gr.Button(
                "Exit — Thoát hẳn (kill tiến trình dịch)", variant="stop"
            )
            tech_details_tog = gr.Markdown(
                tech_details_string,
                elem_classes=["secondary-text"],
            )
            page_range.select(on_select_page, page_range, [page_input, page_range_row])
            service.select(
                on_select_service,
                service,
                envs,
            )
            vfont.change(on_vfont_change, inputs=vfont, outputs=None)
            file_type.select(
                on_select_filetype,
                file_type,
                [file_input, link_input],
                js=(
                    f"""
                    (a,b)=>{{
                        try{{
                            grecaptcha.render('recaptcha-box',{{
                                'sitekey':'{client_key}',
                                'callback':'onVerify'
                            }});
                        }}catch(error){{}}
                        return [a];
                    }}
                    """
                    if flag_demo
                    else ""
                ),
            )

        with gr.Column(scale=2):
            gr.Markdown("## Preview")
            preview = PDF(label="Document Preview", visible=True, height=2000)

    # Event handlers
    file_input.upload(
        lambda x: x,
        inputs=file_input,
        outputs=preview,
        js=(
            f"""
            (a,b)=>{{
                try{{
                    grecaptcha.render('recaptcha-box',{{
                        'sitekey':'{client_key}',
                        'callback':'onVerify'
                    }});
                }}catch(error){{}}
                return [a];
            }}
            """
            if flag_demo
            else ""
        ),
    )

    state = gr.State({"session_id": None, "paused": False, "busy": False})

    translate_inputs = [
        file_type,
        file_input,
        link_input,
        service,
        lang_from,
        lang_to,
        page_range,
        page_input,
        page_from,
        page_to,
        output_format,
        prompt,
        threads,
        skip_subset_fonts,
        ignore_cache,
        vfont,
        use_babeldoc,
        recaptcha_response,
        state,
        *envs,
    ]
    translate_outputs = [
        output_file_mono,
        preview,
        output_file_dual,
        output_file_mono,
        output_file_dual,
        output_title,
    ]

    _recaptcha_reset = "()=>{grecaptcha.reset()}" if flag_demo else ""

    # Translate: shows "Stop" on the dynamic button straight away, then runs.
    translate_btn.click(
        lambda: _toggle_update(False), None, [toggle_btn], queue=False
    )
    translate_btn.click(
        translate_file,
        inputs=translate_inputs,
        outputs=translate_outputs,
    ).then(sync_toggle, [state], [toggle_btn]).then(lambda: None, js=_recaptcha_reset)

    # The dynamic button: label "Stop" -> stops; label "Continue" -> translates
    # again (cache). Unlimited concurrency, otherwise a click on "Stop" would
    # wait in the queue behind a running "Continue".
    toggle_btn.click(
        toggle_translate,
        inputs=[toggle_btn, *translate_inputs],
        outputs=[*translate_outputs, toggle_btn],
        concurrency_limit=None,
    ).then(sync_toggle, [state], [toggle_btn]).then(lambda: None, js=_recaptcha_reset)

    cancellation_btn.click(
        exit_translate_file,
        inputs=[state],
        outputs=[toggle_btn],
    )


def parse_user_passwd(file_path: str) -> tuple:
    """
    Parse the user name and password from the file.

    Inputs:
        - file_path: The file path to read.
    Outputs:
        - tuple_list: The list of tuples of user name and password.
        - content: The content of the file
    """
    tuple_list = []
    content = ""
    if not file_path:
        return tuple_list, content
    if len(file_path) == 2:
        try:
            with open(file_path[1], "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path[1]}' not found.")
    try:
        with open(file_path[0], "r", encoding="utf-8") as file:
            tuple_list = [
                tuple(line.strip().split(",")) for line in file if line.strip()
            ]
    except FileNotFoundError:
        print(f"Error: File '{file_path[0]}' not found.")
    return tuple_list, content


def setup_gui(
    share: bool = False, auth_file: list = ["", ""], server_port=7860
) -> None:
    """
    Setup the GUI with the given parameters.

    Inputs:
        - share: Whether to share the GUI.
        - auth_file: The file path to read the user name and password.

    Outputs:
        - None
    """
    user_list, html = parse_user_passwd(auth_file)
    if flag_demo:
        demo.launch(server_name="0.0.0.0", max_file_size="5mb", inbrowser=True)
    else:
        if len(user_list) == 0:
            try:
                demo.launch(
                    server_name="0.0.0.0",
                    debug=True, show_error=True,
                    inbrowser=True,
                    share=share,
                    server_port=server_port,
                )
            except Exception:
                print(
                    "Error launching GUI using 0.0.0.0.\nThis may be caused by global mode of proxy software."
                )
                try:
                    demo.launch(
                        server_name="127.0.0.1",
                        debug=True, show_error=True,
                        inbrowser=True,
                        share=share,
                        server_port=server_port,
                    )
                except Exception:
                    print(
                        "Error launching GUI using 127.0.0.1.\nThis may be caused by global mode of proxy software."
                    )
                    demo.launch(
                        debug=True, show_error=True, inbrowser=True, share=True, server_port=server_port
                    )
        else:
            try:
                demo.launch(
                    server_name="0.0.0.0",
                    debug=True, show_error=True,
                    inbrowser=True,
                    share=share,
                    auth=user_list,
                    auth_message=html,
                    server_port=server_port,
                )
            except Exception:
                print(
                    "Error launching GUI using 0.0.0.0.\nThis may be caused by global mode of proxy software."
                )
                try:
                    demo.launch(
                        server_name="127.0.0.1",
                        debug=True, show_error=True,
                        inbrowser=True,
                        share=share,
                        auth=user_list,
                        auth_message=html,
                        server_port=server_port,
                    )
                except Exception:
                    print(
                        "Error launching GUI using 127.0.0.1.\nThis may be caused by global mode of proxy software."
                    )
                    demo.launch(
                        debug=True, show_error=True,
                        inbrowser=True,
                        share=True,
                        auth=user_list,
                        auth_message=html,
                        server_port=server_port,
                    )


# For auto-reloading while developing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_gui()