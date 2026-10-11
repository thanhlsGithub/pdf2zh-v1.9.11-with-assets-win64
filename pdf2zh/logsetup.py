"""pdf2zh/logsetup.py

Ghi log ra FILE để khi GUI chỉ hiện "Error" vẫn biết lỗi gì, ở trang nào, ở dòng nào.

Vị trí mặc định:  <thư mục chạy exe>/pdf2zh_logs/
(đổi bằng PDF2ZH_LOG_DIR; nếu không ghi được thì thử ~/.cache/pdf2zh/logs rồi thư mục temp)

  pdf2zh.log   log INFO trở lên + traceback ĐẦY ĐỦ của mọi lỗi (cả tiến trình dịch con / luồng dịch).
               Quá 1 ngày không được ghi thêm thì bị xoá khi mở app lại; file lớn hơn 8 MB
               được xoay thành pdf2zh.log.1 (bản này cũng bị xoá sau 1 ngày).
  crash.log    lỗi "chết cứng" (access violation của onnxruntime / pymupdf / opencv...) do
               faulthandler ghi; nếu GUI báo "process ended unexpectedly" thì xem file này.
  progress.txt MỘT dòng, luôn bị ghi đè: bước đang chạy (vd. "translate page 12/200"). Xong việc thì
               tự xoá; nếu còn sót lại khi mở app thì lần chạy trước đã chết ở bước đó.

Tự dọn (chỉ tiến trình chính, mỗi lần mở app): pdf2zh.log.1 quá 1 ngày bị xoá; crash.log không có
dấu vết crash thật thì làm rỗng, có thì chỉ giữ tối đa 256 KB; progress.txt cũ được đọc rồi xoá.
Tiến trình KHÔNG ghi mỗi trang một dòng log (làm vỡ thanh tiến trình tqdm và phình file log).

Biến môi trường (tuỳ chọn):
  PDF2ZH_LOG_LEVEL   INFO (mặc định) hoặc DEBUG (rất nhiều dòng, dùng khi cần soi kỹ)
  PDF2ZH_LOG_DIR     thư mục chứa log

Cách dùng trong code:
  from pdf2zh.logsetup import setup_logging, report_exception, stage
  setup_logging("gui")                        # gọi nhiều lần cũng được, mỗi tiến trình chỉ cài 1 lần
  detail = report_exception("dịch", exc)      # ghi traceback vào log, trả về chuỗi ngắn để hiện lên GUI
  with stage("OCR prep"): ...                 # ghi thời điểm bắt đầu / xong / lỗi của một công đoạn
  set_progress("translate page 12/200")       # chỉ giữ bước MỚI NHẤT (ghi đè), không thêm dòng log

File này chỉ dùng thư viện chuẩn, không import gì nặng nên import ở đâu cũng an toàn.
"""
import atexit
import faulthandler
import logging
import os
import platform
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

LOG_NAME = "pdf2zh.log"
CRASH_NAME = "crash.log"
PROGRESS_NAME = "progress.txt"
MAX_BYTES = 8 * 1024 * 1024        # pdf2zh.log quá cỡ này thì xoay thành pdf2zh.log.1
BACKUP_DAYS = 1                    # log cũ hơn số ngày này bị xoay / xoá
CRASH_MAX_BYTES = 256 * 1024       # crash.log có crash thật: giữ tối đa chừng này
_CRASH_MARKERS = (
    "Fatal Python error", "Windows fatal exception", "Current thread", "Thread 0x",
    "Stack (most recent call first)",
)
FORMAT = "%(asctime)s %(levelname)-7s [%(process)d %(threadName)s] %(name)s: %(message)s"
# Thư viện ồn ào: chỉ giữ cảnh báo trở lên
_NOISY = (
    "httpx", "httpcore", "http11", "openai", "urllib3", "PIL", "matplotlib",
    "asyncio", "multipart", "watchfiles", "gradio", "python_multipart",
)
# Cảnh báo vô hại nhưng lặp theo từng trang (làm vỡ thanh tiến trình): chỉ giữ lỗi thật
_QUIET = ("pdfminer.pdffont",)
_SECRET_WORDS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD")

_lock = threading.RLock()
_state = {"pid": None, "dir": None, "path": None, "fault_fh": None}
_progress = {"text": "", "t": 0.0}


# --------------------------------------------------------------------------
# Chọn thư mục / file log
# --------------------------------------------------------------------------
def _candidate_dirs():
    env = os.environ.get("PDF2ZH_LOG_DIR", "").strip()
    if env:
        yield Path(env)
    try:
        yield Path.cwd() / "pdf2zh_logs"
    except OSError:
        pass
    yield Path.home() / ".cache" / "pdf2zh" / "logs"
    yield Path(tempfile.gettempdir()) / "pdf2zh_logs"


def _pick_dir() -> Optional[Path]:
    for d in _candidate_dirs():
        try:
            d.mkdir(parents=True, exist_ok=True)
            with open(d / LOG_NAME, "a", encoding="utf-8"):
                pass
            return d
        except OSError:
            continue
    return None


def _is_main_process() -> bool:
    try:
        import multiprocessing

        return multiprocessing.current_process().name == "MainProcess"
    except Exception:
        return True


def _rotate(path: Path) -> None:
    """pdf2zh.log quá BACKUP_DAYS ngày không được ghi: xoá hẳn. Quá MAX_BYTES: xoay thành pdf2zh.log.1."""
    try:
        if not path.exists():
            return
        st = path.stat()
        if time.time() - st.st_mtime > BACKUP_DAYS * 86400:
            path.unlink()                    # quá hạn: xoá hẳn, không giữ bản .1
            return
        if st.st_size > MAX_BYTES:
            old = path.with_name(path.name + ".1")
            if old.exists():
                old.unlink()
            path.rename(old)
    except OSError:
        pass


def _housekeeping(directory: Path) -> str:
    """Dọn file log cũ để không tồn đọng trên máy. Trả về nội dung progress.txt còn sót (nếu có)."""
    now = time.time()
    try:
        old = directory / (LOG_NAME + ".1")
        if old.exists() and now - old.stat().st_mtime > BACKUP_DAYS * 86400:
            old.unlink()
    except OSError:
        pass
    try:
        crash = directory / CRASH_NAME
        if crash.exists():
            text = crash.read_text(encoding="utf-8", errors="replace")
            if not any(m in text for m in _CRASH_MARKERS):
                crash.write_text("", encoding="utf-8")          # chỉ có tiêu đề các lần chạy
            elif len(text.encode("utf-8", errors="replace")) > CRASH_MAX_BYTES:
                crash.write_text(text[-CRASH_MAX_BYTES // 4:], encoding="utf-8")
    except OSError:
        pass
    previous = ""
    try:
        prog = directory / PROGRESS_NAME
        if prog.exists():
            previous = prog.read_text(encoding="utf-8", errors="replace").strip()
            prog.unlink()
    except OSError:
        pass
    return previous


class _Dedup(logging.Filter):
    """Chống lặp: WARNING cùng một mẫu câu chỉ ghi đầy đủ FULL lần đầu (rồi ở các mốc 10/100/1000...);
    ERROR luôn được ghi (từ lần FULL+1 chỉ bỏ phần traceback); CRITICAL và dưới WARNING không đụng tới."""

    FULL = 3
    MILESTONES = (10, 100, 1000, 10000, 100000)

    def __init__(self):
        super().__init__()
        self._seen = {}
        self._lock = threading.Lock()

    def filter(self, record):
        if record.levelno >= logging.CRITICAL or record.levelno < logging.WARNING:
            return True
        key = (record.name, record.levelno, str(record.msg))
        with self._lock:
            n = self._seen[key] = self._seen.get(key, 0) + 1
        if n <= self.FULL:
            return True
        if record.levelno >= logging.ERROR:
            record.exc_info = record.exc_text = None     # giữ dòng báo lỗi, bỏ traceback lặp
            return True
        if n in self.MILESTONES:
            record.msg = f"{record.msg} [lặp lại {n} lần, các lần ở giữa bị ẩn]"
            return True
        return False

    def summary(self):
        with self._lock:
            return [(k, n) for k, n in self._seen.items() if n > self.FULL]


def _log_dedup_summary(dedup: "_Dedup") -> None:
    try:
        lg = logging.getLogger("pdf2zh.logsetup")
        for (name, levelno, msg), n in dedup.summary():
            lg.info("repeated %s x%d [%s]: %s", logging.getLevelName(levelno), n, name, _short(msg, 200))
    except Exception:
        pass


def _level() -> int:
    value = getattr(logging, os.environ.get("PDF2ZH_LOG_LEVEL", "INFO").strip().upper(), None)
    return value if isinstance(value, int) else logging.INFO


# --------------------------------------------------------------------------
# Cài đặt
# --------------------------------------------------------------------------
def setup_logging(role: str = "main") -> Optional[Path]:
    """Gắn file log vào root logger của tiến trình hiện tại. Trả về đường dẫn file log
    (hoặc None nếu không ghi được ở đâu cả). Gọi nhiều lần vẫn chỉ cài 1 lần cho mỗi tiến trình."""
    with _lock:
        if _state["pid"] == os.getpid():
            return _state["path"]
        _state["pid"] = os.getpid()
        directory = _pick_dir()
        if directory is None:
            return None
        path = directory / LOG_NAME
        previous = ""
        if _is_main_process():
            _rotate(path)
            previous = _housekeeping(directory)
        try:
            handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        except OSError:
            return None
        handler.setFormatter(logging.Formatter(FORMAT))
        handler._pdf2zh_file = True  # để nhận ra handler của mình
        dedup = _Dedup()
        handler.addFilter(dedup)
        atexit.register(_log_dedup_summary, dedup)
        level = _level()
        handler.setLevel(level)
        root = logging.getLogger()
        for h in list(root.handlers):  # handler thừa kế từ tiến trình cha (fork)
            if getattr(h, "_pdf2zh_file", False):
                root.removeHandler(h)
        root.addHandler(handler)
        if root.level == logging.NOTSET or root.level > level:
            root.setLevel(level)
        for name in _NOISY:
            logging.getLogger(name).setLevel(logging.WARNING)
        for name in _QUIET:
            logging.getLogger(name).setLevel(logging.ERROR)
        _state["dir"], _state["path"] = directory, path
        _install_hooks(directory)
        _log_header(role)
        if previous:
            logging.getLogger("pdf2zh.logsetup").warning(
                "previous run did not finish; its last step was: %s", previous
            )
        return path


def _install_hooks(directory: Path) -> None:
    prev = sys.excepthook

    def _excepthook(tp, ex, tb):
        if not issubclass(tp, KeyboardInterrupt):
            logging.getLogger("pdf2zh.uncaught").critical(
                "Uncaught exception", exc_info=(tp, ex, tb)
            )
        try:
            prev(tp, ex, tb)
        except Exception:
            pass

    sys.excepthook = _excepthook

    if hasattr(threading, "excepthook"):
        prev_thread = threading.excepthook

        def _thread_hook(args):
            if args.exc_type is not SystemExit:
                logging.getLogger("pdf2zh.uncaught").critical(
                    "Uncaught exception in thread %s",
                    getattr(args.thread, "name", "?"),
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                )
            try:
                prev_thread(args)
            except Exception:
                pass

        threading.excepthook = _thread_hook

    try:  # lỗi chết cứng ở tầng C (onnxruntime, pymupdf, opencv...)
        fh = open(directory / CRASH_NAME, "a", buffering=1, encoding="utf-8")
        fh.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} ---\n")
        fh.flush()
        faulthandler.enable(file=fh, all_threads=True)
        _state["fault_fh"] = fh
    except Exception:
        pass


def _log_header(role: str) -> None:
    lg = logging.getLogger("pdf2zh.logsetup")
    try:
        import multiprocessing

        proc = multiprocessing.current_process().name
    except Exception:
        proc = "?"
    lg.info("===== pdf2zh log: role=%s process=%s pid=%s =====", role, proc, os.getpid())
    lg.info("python %s | %s | cwd=%s", sys.version.split()[0], platform.platform(), os.getcwd())
    versions = []
    try:
        from importlib import metadata

        for dist in ("PyMuPDF", "numpy", "opencv-python", "opencv-python-headless",
                     "pdfminer.six", "onnxruntime", "gradio", "BabelDOC", "tenacity"):
            try:
                versions.append(f"{dist}={metadata.version(dist)}")
            except Exception:
                pass
    except Exception:
        pass
    if versions:
        lg.info("packages: %s", ", ".join(versions))
    env = {
        k: v for k, v in os.environ.items()
        if k.startswith("PDF2ZH_") and not any(w in k.upper() for w in _SECRET_WORDS)
    }
    if env:
        lg.info("env: %s", env)


# --------------------------------------------------------------------------
# Tiến trình: chỉ giữ bước MỚI NHẤT
# --------------------------------------------------------------------------
def set_progress(text: str) -> None:
    """Ghi nhớ bước đang chạy (vd. "translate page 12/200"). Không thêm dòng log nào: chỉ ghi đè
    progress.txt (tối đa 2 lần/giây). Lỗi xảy ra thì báo cáo lỗi kèm bước này."""
    _progress["text"] = text
    directory = _state["dir"]
    now = time.monotonic()
    if directory is None or now - _progress["t"] < 0.5:
        return
    _progress["t"] = now
    try:
        (directory / PROGRESS_NAME).write_text(
            f"{time.strftime('%H:%M:%S')} {text}\n", encoding="utf-8"
        )
    except OSError:
        pass


def fmt_pages(pages, one_based: bool = True) -> str:
    """Rút gọn danh sách trang để ghi log: [0..49] -> "1-50 (50 pages)"; rỗng/None -> "all".
    Sách 1000 trang vẫn chỉ tốn một dòng ngắn."""
    if not pages:
        return "all"
    try:
        nums = sorted({int(p) for p in pages})
    except (TypeError, ValueError):
        return "?"
    off = 1 if one_based else 0
    runs, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        runs.append((start, prev))
        start = prev = n
    runs.append((start, prev))
    parts = [str(a + off) if a == b else f"{a + off}-{b + off}" for a, b in runs]
    if len(parts) > 8:
        parts = parts[:8] + ["..."]
    return f"{','.join(parts)} ({len(nums)} pages)"


def get_progress() -> str:
    return _progress["text"]


def clear_progress() -> None:
    """Xong việc: bỏ bước hiện tại và xoá progress.txt (không để sót file)."""
    _progress["text"] = ""
    _progress["t"] = 0.0
    directory = _state["dir"]
    if directory is not None:
        try:
            (directory / PROGRESS_NAME).unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------
# Truy vấn
# --------------------------------------------------------------------------
def get_log_path() -> Optional[Path]:
    return _state["path"]


def get_crash_path() -> Optional[Path]:
    d = _state["dir"]
    return (d / CRASH_NAME) if d else None


def log_hint() -> str:
    """Câu ngắn chỉ chỗ xem log, để gắn vào thông báo lỗi trên GUI."""
    p = _state["path"]
    return f"Log: {p}" if p else ""


# --------------------------------------------------------------------------
# Báo lỗi
# --------------------------------------------------------------------------
def _short(text, limit: int = 600) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _frame_label(frame) -> str:
    p = Path(frame.filename)
    return f"{p.parent.name}/{p.name}:{frame.lineno} in {frame.name}"


def describe(exc: BaseException, with_hint: bool = True) -> str:
    """Tóm tắt lỗi thành vài dòng: loại lỗi + nội dung, nơi lỗi xảy ra, nơi pdf2zh gọi tới, nguyên nhân gốc."""
    lines = [f"{type(exc).__name__}: {_short(exc)}"]
    try:
        frames = traceback.extract_tb(exc.__traceback__)
    except Exception:
        frames = []
    if frames:
        last = frames[-1]
        own = next(
            (f for f in reversed(frames) if "/pdf2zh/" in f.filename.replace("\\", "/")),
            None,
        )
        lines.append(f"  at: {_frame_label(last)}")
        if own is not None and own is not last:
            lines.append(f"  from: {_frame_label(own)}")
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        lines.append(f"  cause: {type(cause).__name__}: {_short(cause, 300)}")
    if get_progress():
        lines.append(f"  last step: {get_progress()}")
    if with_hint and log_hint():
        lines.append(f"  {log_hint()}")
    return "\n".join(lines)


def report_exception(where: str, exc: Optional[BaseException] = None) -> str:
    """Ghi traceback đầy đủ của `exc` vào log và trả về chuỗi ngắn để hiện lên GUI."""
    exc = exc if exc is not None else sys.exc_info()[1]
    if exc is None:
        return ""
    try:
        logging.getLogger("pdf2zh.error").error(
            "%s failed (last step: %s): %s: %s", where, get_progress() or "-",
            type(exc).__name__, exc, exc_info=(type(exc), exc, exc.__traceback__),
        )
    except Exception:
        pass
    return describe(exc)


@contextmanager
def stage(name: str):
    """Ghi log bắt đầu / xong / lỗi của một công đoạn (traceback đầy đủ do nơi bắt lỗi cuối cùng ghi)."""
    lg = logging.getLogger("pdf2zh.stage")
    t0 = time.monotonic()
    lg.info(">> %s", name)
    try:
        yield
    except BaseException as e:  # noqa: BLE001
        if type(e).__name__ == "CancelledError":
            lg.info("-- %s cancelled after %.1fs", name, time.monotonic() - t0)
            clear_progress()
        else:  # giữ progress.txt: nó cho biết lỗi xảy ra ở bước nào
            lg.error("!! %s failed after %.1fs (last step: %s): %s: %s", name,
                     time.monotonic() - t0, get_progress() or "-", type(e).__name__, e)
        raise
    else:
        lg.info("<< %s done in %.1fs", name, time.monotonic() - t0)
        clear_progress()