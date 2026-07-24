"""Clipboard image → local attachment (for TUI paste).

Terminal paste only carries text. When the OS clipboard holds a bitmap,
we dump it under ``.codedoggy/attachments/`` and return that path so the
TUI can render a chip and submit the same file as structured model input.

Windows notes
-------------
* Win+Shift+S / WeChat / browsers often expose ``PNG`` or ``image/png`` in
  addition to (or instead of) ``CF_DIB``. Reading only ``CF_DIB`` misses many
  real-world screenshots.
* Windows Terminal steals Ctrl+V for "paste text" — image-only clipboards
  produce an empty paste and the app never sees ``c-v``. The TUI polls the
  Ctrl+V chord on Windows and pastes images itself when present.
* True OS drag-and-drop is not delivered to console TUIs; WT may paste the
  dropped file path as text — :func:`coerce_image_path_text` handles that.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ATTACH_DIRNAME = Path(".codedoggy") / "attachments"
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def save_clipboard_image(cwd: Path | str, *, prefix: str = "paste") -> Path | None:
    """If the system clipboard has an image, save a file and return absolute path.

    Returns ``None`` when there is no image (caller should fall back to text paste).
    """
    root = Path(cwd).resolve()
    out_dir = root / _ATTACH_DIRNAME
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.debug("attachments dir create failed: %s", e)
        return None

    dest_png = _next_path(out_dir, prefix=prefix, ext="png")
    saved: Path | None = None
    if sys.platform == "win32":
        saved = _save_windows(dest_png, out_dir=out_dir, prefix=prefix)
    elif sys.platform == "darwin":
        saved = _save_macos(dest_png)
    else:
        saved = _save_linux(dest_png)

    if saved is None or not saved.is_file() or saved.stat().st_size < 32:
        _cleanup_tiny(dest_png)
        _cleanup_tiny(dest_png.with_suffix(".bmp"))
        return None
    return saved.resolve()


def get_system_clipboard_text() -> str | None:
    """Read plain text from the OS clipboard (not prompt_toolkit's internal pad)."""
    if sys.platform == "win32":
        return _windows_clipboard_text()
    if sys.platform == "darwin":
        return _macos_clipboard_text()
    return _linux_clipboard_text()


def set_system_clipboard_text(text: str) -> bool:
    """Write plain text to the OS clipboard."""
    if sys.platform == "win32":
        return _set_windows_clipboard_text(text)
    if sys.platform == "darwin":
        return _set_process_clipboard_text(["pbcopy"], text)
    if shutil.which("wl-copy"):
        return _set_process_clipboard_text(["wl-copy"], text)
    if shutil.which("xclip"):
        return _set_process_clipboard_text(
            ["xclip", "-selection", "clipboard"],
            text,
        )
    return False


def _set_process_clipboard_text(command: list[str], text: str) -> bool:
    try:
        proc = subprocess.run(
            command,
            input=text,
            text=True,
            capture_output=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def _set_windows_clipboard_text(text: str) -> bool:
    """CF_UNICODETEXT writer; ownership transfers to Windows on success."""
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return False

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL

    payload = (text or "") + "\0"
    size = len(payload.encode("utf-16-le"))
    handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
    if not handle:
        return False
    owned_by_windows = False
    try:
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return False
        try:
            source = ctypes.create_unicode_buffer(payload)
            ctypes.memmove(ptr, source, size)
        finally:
            kernel32.GlobalUnlock(handle)

        if not user32.OpenClipboard(None):
            return False
        try:
            if not user32.EmptyClipboard():
                return False
            if not user32.SetClipboardData(CF_UNICODETEXT, handle):
                return False
            owned_by_windows = True
            return True
        finally:
            user32.CloseClipboard()
    finally:
        if not owned_by_windows:
            kernel32.GlobalFree(handle)


def coerce_image_path_text(
    text: str | None, *, cwd: Path | str | None = None
) -> Path | None:
    """If ``text`` is a local image path / file URI, return an existing file.

    Covers Windows Terminal drag-drop (pastes the path as text) and
    Explorer "Copy as path".
    """
    if not text or not str(text).strip():
        return None
    raw = str(text).strip().strip('"').strip("'")
    # file:///C:/foo.png or file://localhost/C:/foo.png
    if raw.lower().startswith("file:"):
        raw = re.sub(r"^file:(?://localhost)?/+", "", raw, flags=re.I)
        if re.match(r"^[A-Za-z]:", raw):
            pass
        else:
            raw = "/" + raw.lstrip("/")
    # One path per paste (first non-empty line).
    line = raw.splitlines()[0].strip().strip('"').strip("'")
    if not line:
        return None
    candidate = Path(line)
    if not candidate.is_absolute() and cwd is not None:
        candidate = Path(cwd) / line
    try:
        candidate = candidate.expanduser()
        if candidate.is_file() and candidate.suffix.lower() in _IMAGE_EXTS:
            return candidate.resolve()
    except OSError:
        return None
    return None


def insert_path_token(path: Path | str, *, cwd: Path | str | None = None) -> str:
    """Text inserted into the prompt buffer (quoted when spaces).

    Prefer a short relative path under ``cwd`` when possible so the chip stays readable.
    """
    p = Path(path).resolve()
    root = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    try:
        text = str(p.relative_to(root))
    except (OSError, ValueError):
        text = str(p)
    text = text.replace("\\", "/")
    if any(ch.isspace() for ch in text):
        return f'"{text}"'
    return text


def insert_image_chip(path: Path | str, *, cwd: Path | str | None = None) -> str:
    """Paste form: visible image chip for Ctrl+click and attachment lookup."""
    from codedoggy.tui.open_path import VIEW_IMAGE_LABEL

    return f"{VIEW_IMAGE_LABEL}({insert_path_token(path, cwd=cwd)})"


def _next_path(out_dir: Path, *, prefix: str, ext: str) -> Path:
    n = 1
    while True:
        candidate = out_dir / f"{prefix}-{n}.{ext}"
        if not candidate.exists():
            return candidate
        n += 1
        if n > 10_000:
            return out_dir / f"{prefix}-{os.getpid()}.{ext}"


def _cleanup_tiny(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size < 32:
            path.unlink()
    except OSError:
        pass


def _copy_into_attachments(
    src: Path, out_dir: Path, *, prefix: str
) -> Path | None:
    ext = src.suffix.lower().lstrip(".") or "png"
    if src.suffix.lower() not in _IMAGE_EXTS:
        ext = "png"
    dest = _next_path(out_dir, prefix=prefix, ext=ext)
    try:
        shutil.copy2(src, dest)
    except OSError as e:
        logger.debug("copy attachment failed: %s", e)
        return None
    if dest.is_file() and dest.stat().st_size >= 32:
        return dest.resolve()
    return None


def _save_windows(
    dest: Path, *, out_dir: Path, prefix: str
) -> Path | None:
    """Prefer raw PNG/JPEG clipboard bytes, then HDROP, DIB, WinForms."""
    # 1) Modern apps: registered PNG / image/* bytes (Win+Shift+S, browsers…).
    for name, ext in (
        ("PNG", "png"),
        ("image/png", "png"),
        ("image/jpeg", "jpg"),
        ("image/jpg", "jpg"),
        ("image/webp", "webp"),
        ("image/bmp", "bmp"),
        ("image/gif", "gif"),
    ):
        blob = _windows_clipboard_format_bytes(name)
        if blob and len(blob) > 32:
            target = dest if ext == "png" else _next_path(out_dir, prefix=prefix, ext=ext)
            try:
                target.write_bytes(blob)
            except OSError:
                continue
            if target.is_file() and target.stat().st_size > 32:
                return target.resolve()

    # 2) Explorer "Copy" / some shots: file drop list.
    for dropped in _windows_hdrop_paths():
        if dropped.suffix.lower() in _IMAGE_EXTS and dropped.is_file():
            copied = _copy_into_attachments(dropped, out_dir, prefix=prefix)
            if copied is not None:
                return copied

    # 3) Classic DIB → BMP (no process spawn).
    dib = _save_windows_dib(dest.with_suffix(".bmp"))

    # 4) WinForms GetImage → PNG (covers CF_BITMAP synthesis).
    png = _save_windows_powershell_png(dest)
    if png is not None:
        if dib is not None and dib != png:
            try:
                dib.unlink()
            except OSError:
                pass
        return png
    return dib


def _windows_clipboard_format_bytes(format_name: str) -> bytes | None:
    """Read a named clipboard format (e.g. ``PNG``, ``image/png``) as raw bytes."""
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
    user32.RegisterClipboardFormatW.restype = wintypes.UINT
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalSize.restype = ctypes.c_size_t

    fmt = user32.RegisterClipboardFormatW(format_name)
    if not fmt:
        return None
    if not user32.OpenClipboard(None):
        return None
    try:
        if not user32.IsClipboardFormatAvailable(fmt):
            return None
        handle = user32.GetClipboardData(fmt)
        if not handle:
            return None
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return None
        try:
            size = int(kernel32.GlobalSize(handle))
            if size < 32:
                return None
            return ctypes.string_at(ptr, size)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def _windows_hdrop_paths() -> list[Path]:
    """Paths from CF_HDROP (Explorer copy / some drag sources mirrored to clip)."""
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return []

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    CF_HDROP = 15

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    shell32.DragQueryFileW.argtypes = [
        wintypes.HANDLE,
        wintypes.UINT,
        wintypes.LPWSTR,
        wintypes.UINT,
    ]
    shell32.DragQueryFileW.restype = wintypes.UINT

    if not user32.OpenClipboard(None):
        return []
    try:
        if not user32.IsClipboardFormatAvailable(CF_HDROP):
            return []
        handle = user32.GetClipboardData(CF_HDROP)
        if not handle:
            return []
        count = int(shell32.DragQueryFileW(handle, 0xFFFFFFFF, None, 0))
        out: list[Path] = []
        buf = ctypes.create_unicode_buffer(1024)
        for i in range(count):
            n = int(shell32.DragQueryFileW(handle, i, buf, 1024))
            if n > 0:
                out.append(Path(buf.value))
        return out
    finally:
        user32.CloseClipboard()


def _save_windows_powershell_png(dest: Path) -> Path | None:
    """System.Windows.Forms clipboard image → PNG (needs STA apartment)."""
    ps = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        "Add-Type -AssemblyName System.Drawing; "
        "$img = [System.Windows.Forms.Clipboard]::GetImage(); "
        "if ($null -eq $img) { exit 2 }; "
        f"$img.Save('{_ps_escape(dest)}', [System.Drawing.Imaging.ImageFormat]::Png); "
        "exit 0"
    )
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-STA",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps,
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("windows clipboard PNG save failed: %s", e)
        return None

    if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 32:
        return dest
    return None


def _save_windows_dib(dest_bmp: Path) -> Path | None:
    """Read CF_DIB via ctypes and write a BMP file."""
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    CF_DIB = 8

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalSize.restype = ctypes.c_size_t

    if not user32.OpenClipboard(None):
        return None
    try:
        if not user32.IsClipboardFormatAvailable(CF_DIB):
            return None
        handle = user32.GetClipboardData(CF_DIB)
        if not handle:
            return None
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return None
        try:
            size = int(kernel32.GlobalSize(handle))
            if size < 40:
                return None
            data = ctypes.string_at(ptr, size)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()

    try:
        bi_size = int.from_bytes(data[0:4], "little")
        if bi_size < 12 or bi_size > 124:
            bi_size = 40
        # biBitCount at offset 14 for BITMAPINFOHEADER
        bit_count = int.from_bytes(data[14:16], "little") if len(data) >= 16 else 0
        clr_used = int.from_bytes(data[32:36], "little") if len(data) >= 36 else 0
        palette_entries = 0
        if bit_count and bit_count <= 8:
            palette_entries = clr_used if clr_used else (1 << bit_count)
        pixel_offset = 14 + bi_size + palette_entries * 4

        file_size = 14 + len(data)
        header = bytearray(14)
        header[0:2] = b"BM"
        header[2:6] = int(file_size).to_bytes(4, "little")
        header[10:14] = int(pixel_offset).to_bytes(4, "little")
        dest_bmp.write_bytes(bytes(header) + data)
    except OSError as e:
        logger.debug("write bmp failed: %s", e)
        return None

    if dest_bmp.is_file() and dest_bmp.stat().st_size > 32:
        return dest_bmp.resolve()
    return None


def _windows_clipboard_text() -> str | None:
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return _windows_clipboard_text_ps()

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    CF_UNICODETEXT = 13

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL

    if not user32.OpenClipboard(None):
        return _windows_clipboard_text_ps()
    try:
        if not user32.IsClipboardFormatAvailable(CF_UNICODETEXT):
            return None
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return None
        try:
            text = ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()

    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return text if text else None


def _windows_clipboard_text_ps() -> str | None:
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-STA",
                "-NonInteractive",
                "-Command",
                "Get-Clipboard -Raw",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    text = (proc.stdout or "").replace("\r\n", "\n").replace("\r", "\n")
    # PowerShell may append trailing newline
    return text if text else None


def _save_macos(dest: Path) -> Path | None:
    pngpaste = shutil.which("pngpaste")
    if not pngpaste:
        return None
    try:
        proc = subprocess.run(
            [pngpaste, str(dest)],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode == 0 and dest.is_file() and dest.stat().st_size > 32:
        return dest
    return None


def _macos_clipboard_text() -> str | None:
    try:
        proc = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout or ""
    return text if text else None


def _save_linux(dest: Path) -> Path | None:
    for cmd in (
        ["wl-paste", "--type", "image/png"],
        ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
    ):
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout and len(proc.stdout) > 32:
            try:
                dest.write_bytes(proc.stdout)
                return dest
            except OSError:
                return None
    return None


def _linux_clipboard_text() -> str | None:
    for cmd in (
        ["wl-paste", "--type", "text"],
        ["xclip", "-selection", "clipboard", "-o"],
    ):
        if not shutil.which(cmd[0]):
            continue
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
    return None


def _ps_escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")
