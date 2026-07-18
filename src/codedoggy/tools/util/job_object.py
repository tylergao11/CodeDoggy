"""Windows Job Object helpers for shell process trees.

Ported intent from Grok bash timeout wording (Job Object kill descendants).
Uses Win32 APIs via ctypes — **real** Job Objects, not a claim over taskkill alone.

Fidelity: **C** — Create/Assign/Terminate/Close; no full Grok cgroup/terminal actor.
Fallback: callers still may use taskkill if Job Object APIs fail.

Source reference (behavior, not line-port of Rust):
  grok-build bash timeout enforcement on Windows → Job Object kill
"""

from __future__ import annotations

import sys
from typing import Any

_job_handles: dict[int, Any] = {}  # pid → job handle (Windows)


def job_objects_supported() -> bool:
    return sys.platform == "win32"


def create_and_assign_job(pid: int) -> bool:
    """Create a Job Object and assign *pid* to it. Returns True on success."""
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

        # Enough rights to assign into a job (and terminate later via the job).
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001
        PROCESS_DUP_HANDLE = 0x0040
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_SUSPEND_RESUME = 0x0800
        SYNCHRONIZE = 0x00100000
        access = (
            PROCESS_SET_QUOTA
            | PROCESS_TERMINATE
            | PROCESS_DUP_HANDLE
            | PROCESS_QUERY_INFORMATION
            | PROCESS_SUSPEND_RESUME
            | SYNCHRONIZE
        )

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

        # Kill-on-close so abandoning the handle still reaps the tree when we CloseHandle
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

    # JOBOBJECT_EXTENDED_LIMIT_INFORMATION is large; use basic limit info path
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
    """Terminate Job Object for *pid* if we created one. Returns True if used."""
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
    """Close job handle without terminate (process already exited)."""
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
    """Shared kill path for bash FG timeout and task_manager.

    Windows: TerminateJobObject if we own a job for *proc.pid*, else taskkill /T.
    POSIX: SIGTERM process group, then SIGKILL after ~1s.
    """
    import os
    import signal
    import subprocess
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
        if terminate_job_for_pid(pid):
            try:
                proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass
        return

    # POSIX
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
