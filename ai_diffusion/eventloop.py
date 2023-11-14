import asyncio
from PyQt5.QtCore import QTimer

_loop = asyncio.new_event_loop()
_timer = QTimer()


def process_python_events():
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
