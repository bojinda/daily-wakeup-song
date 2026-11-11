"""Same-thread HTTP reads with one absolute deadline, including header parsing.

urllib's timeout alone is per socket operation. Feed HTTPResponse through a
deadline-aware raw reader so trickled headers, chunks or bodies cannot renew it.
No request workers, timer threads, retries or changes to POST transports.
"""
from functools import partial
import http.client
import io
import ipaddress
import json
import socket
import subprocess
import sys
import time
import urllib.request


def _remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError('HTTP request deadline')
    return value


class _Reader(io.RawIOBase):
    def __init__(self, sock, deadline):
        super().__init__()
        self.sock, self.deadline = sock, deadline
        self.raw = sock.makefile('rb', buffering=0)

    def readable(self):
        return True

    def readinto(self, buffer):
        self.sock.settimeout(_remaining(self.deadline))
        count = self.raw.readinto(buffer)
        _remaining(self.deadline)
        return count

    def close(self):
        try:
            self.raw.close()
        finally:
            super().close()


class _ResponseSocket:
    def __init__(self, sock, deadline):
        self.sock, self.deadline = sock, deadline

    def makefile(self, mode):
        assert mode == 'rb'
        return io.BufferedReader(_Reader(self.sock, self.deadline))


def _response(deadline, sock, *args, **kwargs):
    return http.client.HTTPResponse(_ResponseSocket(sock, deadline), *args, **kwargs)


def _addresses(host, port, deadline):
    try:
        ipaddress.ip_address(host)
    except ValueError:
        # getaddrinfo has no cancellation API. Only hostname resolution uses a
        # short-lived, kill/reaped helper; no HTTP, credentials or inference in
        # it. Its startup and resolution share the request's remaining budget.
        script = ('import json,socket,sys; '
                  'print(json.dumps(socket.getaddrinfo(sys.argv[1],int(sys.argv[2]),'
                  '0,socket.SOCK_STREAM)[:64]))')
        result = subprocess.run([sys.executable, '-I', '-c', script, host, str(port)],
            check=True, capture_output=True, timeout=_remaining(deadline),
            **({'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {}))
        addresses = json.loads(result.stdout)
    else:
        addresses = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM, 0, socket.AI_NUMERICHOST)
    _remaining(deadline)
    return addresses


def _connect(deadline, address, timeout=None, source_address=None):
    last_error = None
    for family, kind, proto, _, destination in _addresses(*address, deadline):
        sock = socket.socket(family, kind, proto)
        try:
            sock.settimeout(_remaining(deadline))
            if source_address:
                sock.bind(source_address)
            sock.connect(tuple(destination))
            # HTTPSConnection immediately performs its certificate-verified TLS
            # handshake. Give that operation only the time left after connect.
            sock.settimeout(_remaining(deadline))
            return sock
        except OSError as exc:
            sock.close()
            last_error = exc
    if last_error is not None:
        raise last_error
    raise OSError('No backend addresses')


class _DeadlineConnection:
    def connect(self):
        super().connect()
        self.sock.settimeout(_remaining(self.deadline))

    def send(self, data):
        if self.sock is not None:
            self.sock.settimeout(_remaining(self.deadline))
        super().send(data)
        _remaining(self.deadline)


class _HTTPConnection(_DeadlineConnection, http.client.HTTPConnection):
    pass


class _HTTPSConnection(_DeadlineConnection, http.client.HTTPSConnection):
    pass


def _connection(deadline, cls, *args, **kwargs):
    kwargs['timeout'] = _remaining(deadline)
    connection = cls(*args, **kwargs)
    connection.deadline = deadline
    connection._create_connection = partial(_connect, deadline)
    connection.response_class = partial(_response, deadline)
    return connection


class _HTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline

    def http_open(self, request):
        return self.do_open(partial(_connection, self.deadline, _HTTPConnection), request)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline

    def https_open(self, request):
        return self.do_open(partial(_connection, self.deadline, _HTTPSConnection), request,
                            context=self._context)


def read_opener(deadline, redirect_handler):
    # Retain urllib's validation, certificate verification and HTTP error
    # handling. Explicit empty proxies and the original redirect refusal apply.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), redirect_handler,
                                      _HTTPHandler(deadline), _HTTPSHandler(deadline))
