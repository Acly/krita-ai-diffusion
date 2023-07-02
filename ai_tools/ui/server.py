from enum import Enum
from typing import Optional
from PyQt5.QtCore import QObject, pyqtSignal
from .. import Auto1111, eventloop, settings


class ServerState(Enum):
    disconnected = 0
    connecting = 1
    connected = 2
    error = 3


class DiffusionServer(QObject):
    """ViewModel for the diffusion server connection."""

    changed = pyqtSignal()

    state = ServerState.disconnected
    diffusion: Optional[Auto1111] = None
    error: Optional[str] = None

    def __init__(self):
        super().__init__()

    async def _connect(self):
        self.state = ServerState.connecting
        self.error = None
        self.changed.emit()
        try:
            self.diffusion = await Auto1111.connect(settings.server_url)
            self.state = ServerState.connected
        except Exception as e:
            self.error = str(e)
            self.state = ServerState.error
        self.changed.emit()

    def connect(self):
        eventloop.run(self._connect())

    def interrupt(self):
        eventloop.run(self.diffusion.interrupt())


# Global view model so diffusion widgets created by Krita can access it
diffusion_server = DiffusionServer()
