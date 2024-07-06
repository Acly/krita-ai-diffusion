import asyncio
from typing import Callable
from PyQt5.QtCore import QTimer

_loop = asyncio.new_event_loop()
_timer = QTimer()


def process_python_events():
    if not _loop.is_running():
        _loop.call_soon(lambda: _loop.stop())
        _loop.run_forever()


def setup():
    assert _timer is not None
    _timer.setInterval(20)
    _timer.timeout.connect(process_python_events)
    _timer.start()


def run(future):
    task = None

    def schedule():
        nonlocal task
        task = _loop.create_task(future)
        _loop.stop()

    if not _loop.is_running():
        _loop.call_soon(schedule)
        _loop.run_forever()
    else:
        task = _loop.create_task(future)
    assert task, "Task was not scheduled"
    return task


def stop():
    global _timer, _loop
    try:
        _timer.stop()
        _timer = None
        _loop.stop()
        _loop.close()
    except Exception:
        pass


async def wait_until(condition: Callable[[], bool], iterations=10, no_error=False):
    while not condition() and iterations > 0:
        iterations -= 1
        if iterations == 0 and not no_error:
            raise TimeoutError("Timeout while waiting for action to complete")
        await asyncio.sleep(0.01)


async def process_events():
    # This is usually a hack where some API requires certain events to be processed first
    # and this is not enforced by Krita
    await asyncio.sleep(0.001)
