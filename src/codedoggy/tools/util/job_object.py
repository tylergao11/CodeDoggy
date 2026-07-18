"""Windows Job Object helpers for shell process trees.

Ported from Grok (behavior, not line-for-line Rust):
  crates/codegen/xai-tty-utils/src/lib.rs  ProcessGroup (Windows)
    CreateJobObjectW + JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    AssignProcessToJobObject
    TerminateJobObject
  crates/codegen/xai-grok-tools/src/computer/local/terminal.rs
    send_sigkill_to_group: ProcessGroup.kill() **then** child.start_kill()
    (both steps always; not a degrade ladder — Grok has no taskkill)

Project rule: do **not** invent silent alternate kill mechanisms.
"""

from __future__ import annotations

import sys
from typing import Any

_job_handles: dict[int, Any] = {}  # pid → job handle (Windows)


def job_objects_supported() -> bool:
    return sys.platform == "win32"


def create_and_assign_job(pid: int) -> bool:
    """Create a Job Object and assign *pid* to it. Returns True on success.

    Grok ``ProcessGroup::new`` + ``attach_pid`` spirit.
    """
    if not job_objects_supported():
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        CreateJobObjectW = kernel32.CreateJobObjectW
        CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        CreateJobObjectW.restype = wintypes.HANDLE

        AssignProcessToJobObject = kernel32.AssignProcessToJobObject
        AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        AssignProcessToJobObject.restype = wintypes.BOOL

        OpenProcess = kernel32.OpenProcess
        OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        OpenProcess.restype = wintypes.HANDLE

        CloseHandle = kernel32.CloseHandle
        CloseHandle.argtypes = [wintypes.HANDLE]
        CloseHandle.restype = wintypes.BOOL

        # Grok OpenProcess: PROCESS_SET_QUOTA | PROCESS_TERMINATE
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001
        access = PROCESS_SET_QUOTA | PROCESS_TERMINATE

        job = CreateJobObjectW(None, None)
        if not job:
            return False

        hproc = OpenProcess(access, False, int(pid))
        if not hproc:
            CloseHandle(job)
            return False
        try:
            ok = AssignProcessToJobObject(job, hproc)
            if not ok:
                CloseHandle(job)
                return False
        finally:
            CloseHandle(hproc)

        # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (Grok SetInformationJobObject)
        try:
            _set_kill_on_job_close(kernel32, job)
        except Exception:  # noqa: BLE001
            pass

        _job_handles[int(pid)] = job
        return True
    except Exception:  # noqa: BLE001
        return False


def _set_kill_on_job_close(kernel32: Any, job: Any) -> None:
    import ctypes
    from ctypes import wintypes

    JobObjectExtendedLimitInformation = 9
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    SetInformationJobObject = kernel32.SetInformationJobObject
    SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    SetInformationJobObject.restype = wintypes.BOOL
    SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )


def terminate_job_for_pid(pid: int, exit_code: int = 1) -> bool:
    """Terminate Job Object for *pid* if we created one. Returns True if used.

    Grok ``ProcessGroup::terminate_job`` / ``TerminateJobObject``.
    """
    if not job_objects_supported():
        return False
    job = _job_handles.pop(int(pid), None)
    if job is None:
        return False
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        TerminateJobObject = kernel32.TerminateJobObject
        TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        TerminateJobObject.restype = wintypes.BOOL
        CloseHandle = kernel32.CloseHandle
        CloseHandle.argtypes = [wintypes.HANDLE]
        CloseHandle.restype = wintypes.BOOL
        TerminateJobObject(job, int(exit_code))
        CloseHandle(job)
        return True
    except Exception:  # noqa: BLE001
        return False


def release_job_for_pid(pid: int) -> None:
    """Close job handle without terminate (process already exited).

    Grok Drop of ProcessGroup closes the job handle (KILL_ON_JOB_CLOSE may
    still reap remaining members).
    """
    job = _job_handles.pop(int(pid), None)
    if job is None:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        CloseHandle = kernel32.CloseHandle
        CloseHandle.argtypes = [wintypes.HANDLE]
        CloseHandle.restype = wintypes.BOOL
        CloseHandle(job)
    except Exception:  # noqa: BLE001
        pass


def kill_process_tree(proc: Any) -> None:
    """Shared kill path — Grok ``send_sigkill_to_group`` (exact step order).

    Windows (terminal.rs hard-kill, both steps always):
      1. ProcessGroup.kill → TerminateJobObject when we hold a job for pid
      2. child.start_kill → proc.kill() on the immediate child

    POSIX:
      1. killpg SIGTERM, then SIGKILL after ~1s
      2. proc.kill() (same “always also kill child” step as Grok)
    """
    import os
    import signal
    import time

    if proc is None:
        return
    if proc.poll() is not None:
        if job_objects_supported():
            try:
                release_job_for_pid(proc.pid)
            except Exception:  # noqa: BLE001
                pass
        return

    pid = proc.pid
    if job_objects_supported():
        # Grok: process_group.kill() then child.start_kill()
        terminate_job_for_pid(pid)
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=2.0)
        except Exception:  # noqa: BLE001
            pass
        return

    # POSIX — Grok killpg SIGTERM / SIGKILL + child start_kill spirit
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.kill()
    except OSError:
        pass
