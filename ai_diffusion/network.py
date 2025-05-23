from __future__ import annotations
import asyncio
import json
import os
from asyncio import Future
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from PyQt5.QtCore import QByteArray, QUrl, QFile, QBuffer
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply, QSslError

from .localization import translate as _
from .util import client_logger as log


class NetworkError(Exception):
    code: int
    message: str
    url: str
    status: int | None = None
    data: dict | None = None

    def __init__(
        self, code: int, msg: str, url: str, status: int | None = None, data: dict | None = None
    ):
        self.code = code
        self.message = msg
        self.url = url
        self.status = status
        self.data = data
        super().__init__(self, msg)

    def __str__(self):
        return self.message

    @staticmethod
    def from_reply(reply: QNetworkReply):
        code: QNetworkReply.NetworkError = reply.error()  # type: ignore (bug in PyQt5-stubs)
        url = reply.url().toString()
        status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        if reply.isReadable():
            try:  # extract detailed information from the payload
                data = json.loads(reply.readAll().data())
                error = data.get("error", "Network error")
                return NetworkError(code, f"{error} ({reply.errorString()})", url, status, data)
            except Exception:
                try:
                    text = reply.readAll().data().decode("utf-8")
                    if text:
                        return NetworkError(code, f"{text} ({reply.errorString()})", url, status)
                except Exception:
                    pass
        if code == QNetworkReply.NetworkError.OperationCanceledError:
            return NetworkError(
                code, "Connection timed out, the server took too long to respond", url
            )
        return NetworkError(code, reply.errorString(), url, status)


class OutOfMemoryError(NetworkError):
    def __init__(self, code, msg, url):
        super().__init__(code, msg, url)


class Disconnected(Exception):
    def __init__(self):
        super().__init__(self, "Disconnected")


class Request(NamedTuple):
    url: str
    future: asyncio.Future
    buffer: QBuffer | None = None


Headers = list[tuple[str, str]]


class RequestManager:
    def __init__(self):
        self._net = QNetworkAccessManager()
        self._net.finished.connect(self._finished)
        self._net.sslErrors.connect(self._handle_ssl_errors)
        self._requests: dict[QNetworkReply, Request] = {}
        self._upload_future: Future[tuple[int, int]] | None = None

    def http(
        self,
        method,
        url: str,
        data: dict | QByteArray | None = None,
        bearer="",
        headers: Headers | None = None,
        timeout: float | None = None,
    ):
        self._cleanup()

        request = QNetworkRequest(QUrl(url))
        request.setAttribute(QNetworkRequest.FollowRedirectsAttribute, True)
        request.setRawHeader(b"ngrok-skip-browser-warning", b"69420")
        if bearer:
            request.setRawHeader(b"Authorization", f"Bearer {bearer}".encode("utf-8"))
        if headers:
            for key, value in headers:
                request.setRawHeader(key.encode("utf-8"), value.encode("utf-8"))
        if timeout is not None:
            request.setTransferTimeout(int(timeout * 1000))

        assert method in ["GET", "POST", "PUT"]
        if method == "POST":
            data = data or {}
            data_bytes = QByteArray(json.dumps(data).encode("utf-8"))
            request.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json")
            request.setHeader(QNetworkRequest.KnownHeaders.ContentLengthHeader, data_bytes.size())
            reply = self._net.post(request, data_bytes)
        elif method == "PUT":
            if isinstance(data, bytes):
                data = QByteArray(data)
            assert isinstance(data, QByteArray)
            request.setHeader(
                QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/octet-stream"
            )
            request.setHeader(QNetworkRequest.KnownHeaders.ContentLengthHeader, data.size())
            reply = self._net.put(request, data)
        else:
            reply = self._net.get(request)

        assert reply is not None, f"Network request for {url} failed: reply is None"
        future = asyncio.get_running_loop().create_future()
        self._requests[reply] = Request(url, future)
        return future

    def get(self, url: str, bearer="", timeout: float | None = None):
        return self.http("GET", url, bearer=bearer, timeout=timeout)

    def post(self, url: str, data: dict, bearer=""):
        return self.http("POST", url, data, bearer=bearer)

    def put(self, url: str, data: QByteArray | bytes):
        return self.http("PUT", url, data)

    async def upload(self, url: str, data: QByteArray | bytes, sha256: str | None = None):
        self._cleanup()
        if isinstance(data, bytes):
            data = QByteArray(data)
        assert isinstance(data, QByteArray)

        request = QNetworkRequest(QUrl(url))
        request.setAttribute(QNetworkRequest.Attribute.FollowRedirectsAttribute, True)
        if sha256:
            request.setRawHeader(b"x-amz-checksum-sha256", sha256.encode("utf-8"))
        request.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/octet-stream"
        )
        request.setHeader(QNetworkRequest.KnownHeaders.ContentLengthHeader, data.size())
        reply = self._net.put(request, data)
        assert reply is not None, f"Network request for {url} failed: reply is None"

        reply.uploadProgress.connect(self._upload_progress)
        self._upload_future = asyncio.get_running_loop().create_future()
        finished_future = asyncio.get_running_loop().create_future()
        self._requests[reply] = Request(url, finished_future)
        while self._upload_future is not None:
            fut = next(asyncio.as_completed([self._upload_future, finished_future]))
            progress = await fut
            if isinstance(progress, tuple) and progress[0] != progress[1]:
                self._upload_future = asyncio.get_running_loop().create_future()
                yield progress
            else:
                yield (len(data), len(data))
                break

    def download(self, url: str):
        self._cleanup()
        request = QNetworkRequest(QUrl(url))
        request.setAttribute(QNetworkRequest.Attribute.FollowRedirectsAttribute, True)
        reply = self._net.get(request)
        assert reply is not None, f"Network request for {url} failed: reply is None"

        buffer = QBuffer()
        buffer.open(QBuffer.OpenModeFlag.WriteOnly)

        def write(bytes_received, bytes_total):
            buffer.write(reply.readAll())

        future = asyncio.get_running_loop().create_future()
        tracker = Request(url, future, buffer)
        reply.downloadProgress.connect(write)
        self._requests[reply] = tracker
        return future

    def _upload_progress(self, bytes_sent: int, bytes_total: int):
        if bytes_total == 0:
            return
        if self._upload_future is None or self._upload_future.done():
            return
        if not self._upload_future.cancelled():
            self._upload_future.set_result((bytes_sent, bytes_total))
        self._upload_future = None

    def _finished(self, reply: QNetworkReply):
        future = None
        try:
            code = reply.error()  # type: ignore (bug in PyQt5-stubs)
            tracker = self._requests[reply]
            future = tracker.future
            if future.cancelled():
                return  # operation was cancelled, discard result
            if code == QNetworkReply.NetworkError.NoError:
                if tracker.buffer is not None:
                    tracker.buffer.write(reply.readAll())
                    future.set_result(tracker.buffer.data())
                else:
                    content_type = reply.header(QNetworkRequest.KnownHeaders.ContentTypeHeader)
                    data = reply.readAll().data()
                    if content_type and ("application/json" in content_type):
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

    def _handle_ssl_errors(self, reply: QNetworkReply, errors: list[QSslError]):
        for error in errors:
            log.warning(f"SSL error: {error.errorString()} [{error.error()}]")


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
        raise Exception(
            _("Error during download: could not open {path} for writing", path=out_file.fileName())
        )

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
                raise NetworkError(e.code, _("Failed to download") + f" {url}: {e.message}", url)
        except Exception as e:
            raise Exception(_("Failed to download") + f" {url}: {e}") from e

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
