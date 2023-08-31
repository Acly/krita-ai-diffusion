import asyncio
from PyQt5.QtCore import QTimer

_loop =  asyncio.new_event_loop()
_timer = QTimer()

def process_python_events():
    global _loop
    _loop.call_soon(lambda: _loop.stop())
    _loop.run_forever()

def setup():
    global _timer
    _timer.setInterval(20)
    _timer.timeout.connect(process_python_events)
    _timer.start()

def run(future):
    task = None
    def schedule():
        nonlocal task
        task = _loop.create_task(future)
        _loop.stop()
    _loop.call_soon(schedule)
    _loop.run_forever()
    assert task, 'Task was not scheduled'
    return task
