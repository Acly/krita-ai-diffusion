import ctypes
from ctypes import byref, sizeof
import functools


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
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


JobObjectExtendedLimitInformation = 9

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100


class Kernel32:
    dll = ctypes.windll.kernel32  # type: ignore

    def _kernel32_call(self, func: str, *args):
        result = getattr(self.dll, func)(*args)
        if result == 0:
            raise RuntimeError(f"{func} failed: {self.dll.GetLastError()}")
        return result

    def __getattr__(self, name: str):
        return functools.partial(self._kernel32_call, name)


kernel32 = Kernel32()


def attach_process_to_job(pid):
    """Creates a job object and attaches a newly created process, making sure that it
    is terminated when the parent process terminates (even upon kill/crash).
    """
    job = kernel32.CreateJobObjectA(None, b"")
    ext = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    kernel32.QueryInformationJobObject(
        job, JobObjectExtendedLimitInformation, byref(ext), sizeof(ext), None
    )
    ext.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    kernel32.SetInformationJobObject(
        job, JobObjectExtendedLimitInformation, byref(ext), sizeof(ext)
    )
    process = kernel32.OpenProcess(PROCESS_TERMINATE | PROCESS_SET_QUOTA, False, pid)
    kernel32.AssignProcessToJobObject(job, process)
    kernel32.CloseHandle(process)
