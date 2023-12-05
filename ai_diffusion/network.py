from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Callable
from PyQt5.QtCore import QByteArray, QUrl, QFile
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply

from .util import client_logger as log


class NetworkError(Exception):
    def __init__(self, code, msg, url):
        self.code = code
        self.message = msg
        self.url = url
        super().__init__(self, msg)

    def __str__(self):
        return self.message

    @staticmethod
    def from_reply(reply: QNetworkReply):
        code = reply.error()  # type: ignore (bug in PyQt5-stubs)
        url = reply.url().toString()
        try:  # extract detailed information from the payload
            data = json.loads(reply.readAll().data())
            error = data.get("error", "")
            if error != "":
                return NetworkError(code, f"{error} ({reply.errorString()})", url)
        except:
            pass
        return NetworkError(code, reply.errorString(), url)


class OutOfMemoryError(NetworkError):
    def __init__(self, code, msg, url):
        super().__init__(code, msg, url)


class Interrupted(Exception):
    def __init__(self):
        super().__init__(self, "Operation cancelled")


class Disconnected(Exception):
    def __init__(self):
        super().__init__(self, "Disconnected")


class Request(NamedTuple):
    url: str
    future: asyncio.Future


class RequestManager:
    def __init__(self):
        self._net = QNetworkAccessManager()
        self._net.finished.connect(self._finished)
        self._requests = {}

    def http(self, method, url: str, data: dict | None = None):
        self._cleanup()

        request = QNetworkRequest(QUrl(url))
        # request.setTransferTimeout({"GET": 30000, "POST": 0}[method]) # requires Qt 5.15 (Krita 5.2)
        request.setRawHeader(b"ngrok-skip-browser-warning", b"69420")

        assert method in ["GET", "POST"]
        if method == "POST":
            data = data or {}
            data_bytes = QByteArray(json.dumps(data).encode("utf-8"))
            request.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
            request.setHeader(QNetworkRequest.ContentLengthHeader, data_bytes.size())
            reply = self._net.post(request, data_bytes)
        else:
            reply = self._net.get(request)

        future = asyncio.get_running_loop().create_future()
        self._requests[reply] = Request(url, future)
        return future

    def get(self, url: str):
        return self.http("GET", url)

    def post(self, url: str, data: dict):
        return self.http("POST", url, data)

    def _finished(self, reply: QNetworkReply):
        future = None
        try:
            code = reply.error()  # type: ignore (bug in PyQt5-stubs)
            future = self._requests[reply].future
            if future.cancelled():
                return  # operation was cancelled, discard result
            if code == QNetworkReply.NetworkError.NoError:
                content_type = reply.header(QNetworkRequest.ContentTypeHeader)
                data = reply.readAll().data()
                if "application/json" in content_type:
                    future.set_result(json.loads(data))
                else:
                    future.set_result(data)
            else:
                future.set_exception(NetworkError.from_reply(reply))
        except Exception as e:
            if future is not None:
                future.set_exception(e)

    def _cleanup(self):
        self._requests = {
            reply: request for reply, request in self._requests.items() if not reply.isFinished()
        }


class DownloadProgress(NamedTuple):
    received: float  # in mb
    total: float  # in mb
    speed: float  # in mb/s
    value: float  # in [0, 1]


class DownloadHelper:
    _initial = 0
    _total = 0
    _received = 0
    _time: datetime | None = None

    def __init__(self, resume_from: int = 0):
        self._initial = resume_from / 10**6

    def update(self, received_bytes: int, total_bytes: int = 0):
        received = received_bytes / 10**6
        total = total_bytes / 10**6
        diff = received - self._received
        now = datetime.now()
        speed = 0
        self._received = received
        self._total = max(self._total, total)
        if self._time is not None:
            speed = diff / max((now - self._time).total_seconds(), 0.0001)
        self._time = now
        current = self._initial + self._received
        total = 0
        progress = -1
        if self._total > 0:
            total = self._initial + self._total
            progress = current / total
        return DownloadProgress(current, total, speed, progress)

    def final(self):
        return DownloadProgress(self._initial + self._received, self._initial + self._total, 0, 1)


async def _try_download(network: QNetworkAccessManager, url: str, path: Path):
    out_file = QFile(str(path) + ".part")
    if not out_file.open(QFile.ReadWrite | QFile.Append):  # type: ignore
        raise Exception(f"Error during download: could not open {path} for writing")

    request = QNetworkRequest(QUrl(_map_host(url)))
    request.setAttribute(QNetworkRequest.FollowRedirectsAttribute, True)
    if out_file.size() > 0:
        log.info(f"Found {path}.part, resuming download from {out_file.size()} bytes")
        request.setRawHeader(b"Range", f"bytes={out_file.size()}-".encode("utf-8"))
    reply = network.get(request)
    assert reply is not None, f"Network request for {url} failed: reply is None"

    progress_future = asyncio.get_running_loop().create_future()
    finished_future = asyncio.get_running_loop().create_future()
    progress_helper = DownloadHelper(resume_from=out_file.size())

    def handle_progress(bytes_received, bytes_total):
        out_file.write(reply.readAll())
        result = progress_helper.update(bytes_received, bytes_total)
        if not progress_future.done():
            progress_future.set_result(result)

    def handle_finished():
        out_file.write(reply.readAll())
        out_file.close()
        if finished_future.cancelled():
            return  # operation was cancelled, discard result
        if reply.error() == QNetworkReply.NetworkError.NoError:  # type: ignore (bug in PyQt5-stubs)
            finished_future.set_result(path)
        elif reply.attribute(QNetworkRequest.HttpStatusCodeAttribute) == 416:
            # 416 = Range Not Satisfiable
            finished_future.set_exception(NetworkError(416, "Resume not supported", url))
        else:
            finished_future.set_exception(NetworkError.from_reply(reply))

    reply.downloadProgress.connect(handle_progress)
    reply.finished.connect(handle_finished)

    while not reply.isFinished():
        await asyncio.wait([progress_future, finished_future], return_when=asyncio.FIRST_COMPLETED)
        if progress_future.done():
            progress = progress_future.result()
            progress_future = asyncio.get_running_loop().create_future()
            yield progress

    if e := finished_future.exception():
        raise e

    out_file.rename(str(path))
    yield progress_helper.final()


async def download(network: QNetworkAccessManager, url: str, path: Path):
    for retry in range(3, 0, -1):
        try:
            async for progress in _try_download(network, url, path):
                yield progress
            break
        except NetworkError as e:
            if e.code == 416:  # Range Not Satisfiable
                log.info("Download received code 416: Resume not supported, restarting")
                QFile.remove(str(path) + ".part")
            elif e.code in [
                QNetworkReply.NetworkError.RemoteHostClosedError,
                QNetworkReply.NetworkError.TemporaryNetworkFailureError,
            ]:
                log.warning(f"Download interrupted: {e}")
                if retry == 1:
                    raise e
                await asyncio.sleep(1)
            else:
                raise NetworkError(e.code, f"Failed to download {url}: {e.message}", url)
        except Exception as e:
            raise Exception(f"Failed to download {url}: {e}") from e

        log.info(f"Retrying download of {url}, {retry - 1} attempts left")


HOSTMAP_LOCAL = {  # for testing
    "https://huggingface.co": "http://localhost:51222",
    "https://civitai.com": "http://localhost:51222",
}
HOSTMAP = HOSTMAP_LOCAL if os.environ.get("HOSTMAP") else {}


def _map_host(url: str):
    for host, mapped in HOSTMAP.items():
        if url.startswith(host):
            return mapped + url[len(host) :]
    return url
