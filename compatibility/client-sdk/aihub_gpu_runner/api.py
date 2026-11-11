"""Authenticated asynchronous job API; deliberately not an Ollama gateway."""
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .core import Conflict, encode, strict_json

class Application:
    def __init__(self, runner, token, max_request_bytes=262144):
        if not isinstance(token, str) or len(token) < 16:
            raise ValueError("internal_token_required")
        self.runner, self.token, self.max_request_bytes = runner, token, max_request_bytes
    def authenticated(self, headers):
        authorization = next((v for k, v in headers.items() if k.lower() == "authorization"), "")
        return isinstance(authorization, str) and hmac.compare_digest(
            authorization.encode(), ("Bearer " + self.token).encode())
    def handle(self, method, path, headers, body=b""):
        if not self.authenticated(headers):
            return 401, "application/json", encode({"error": "unauthorized"})
        if len(body) > self.max_request_bytes:
            return 413, "application/json", encode({"error": "request_too_large"})
        try:
            if method == "POST" and path == "/jobs":
                value = strict_json(body)
                if not isinstance(value, dict) or set(value) != {"request_id", "target", "operation", "payload"}:
                    raise ValueError("invalid_envelope")
                job = self.runner.submit(**value)
                return 202, "application/json", encode(self._status(job))
            if method == "GET" and path == "/resources":
                state = self.runner.admission.snapshot()
                return 200, "application/json", encode({"ready": state["ready"],
                    "owners": {r: {"job_id": o["job_id"], "phase": o["phase"]}
                               for r, o in state["owners"].items()}})
            if method == 'GET' and path == '/recovery-summary':
                from .recovery_summary import read_recovery_summary
                return 200, 'application/json', encode(read_recovery_summary(self.runner.store))
            if method == 'GET' and path == '/recovery-incidents':
                from .recovery_summary import read_recovery_summary
                return 200, 'application/json', encode(read_recovery_summary(self.runner.store, include_incidents=True))
            parts = path.split("/")
            if len(parts) in (3, 4) and parts[:2] == ["", "jobs"]:
                job_id = parts[2]
                if len(parts) == 3 and method == "GET":
                    return 200, "application/json", encode(self._status(self.runner.status(job_id)))
                if len(parts) == 4 and parts[3] == "result" and method == "GET":
                    return 200, "application/json", encode(self.runner.result(job_id))
                if len(parts) == 4 and parts[3] == "artifact" and method == "GET":
                    data, media_type = self.runner.artifact(job_id)
                    return 200, media_type, data
                if len(parts) == 4 and parts[3] == "cancel" and method == "POST":
                    cancelled = self.runner.cancel(job_id)
                    return (200 if cancelled else 409), "application/json", encode({"cancelled": cancelled})
            return 404, "application/json", encode({"error": "not_found"})
        except (ValueError, UnicodeError):
            return 400, "application/json", encode({"error": "invalid_request"})
        except KeyError:
            return 404, "application/json", encode({"error": "not_found"})
        except Conflict:
            return 409, "application/json", encode({"error": "conflict_or_result_pending"})
        except Exception:
            return 503, "application/json", encode({"error": "state_unavailable"})
    def _status(self, job):
        future = self.runner.futures.get(job["request_id"])
        active = future is not None and not future.done()
        state = self.runner.admission.snapshot()
        released = state["ready"] and not active and not any(
            owner["job_id"] == job["request_id"] for owner in state["owners"].values())
        result = {key: job[key] for key in ("request_id", "request_hash", "target", "state",
                "created_at", "updated_at", "error_code", "phase", "backend_detail",
                "completion_verified", "cleanup_verified", "publication_warning") if key in job}
        if job.get('wallpaper_lifecycle') and result.get('state') == 'success' and not released:
            result['state'] = 'cleanup_pending'
        return dict(result, resources_released=released, supervision_active=active)

def serve(application, host="127.0.0.1", port=8790):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def process(self):
            self.connection.settimeout(10)
            headers = dict(self.headers)
            if not application.authenticated(headers):
                status, media, data = application.handle(self.command, self.path, headers)
            else:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 0 or length > application.max_request_bytes:
                        status, media, data = 413, "application/json", encode({"error": "request_too_large"})
                    else:
                        body = self.rfile.read(length)
                        if len(body) != length:
                            raise ValueError("truncated_body")
                        status, media, data = application.handle(self.command, self.path, headers, body)
                except (ValueError, TimeoutError, OSError):
                    status, media, data = 400, "application/json", encode({"error": "invalid_body"})
            try:
                self.send_response(status)
                self.send_header("Content-Type", media)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
        do_GET = process
        do_POST = process
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
