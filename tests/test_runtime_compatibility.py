"""Offline tests of the actual Flask application and preserved cooperative SDK."""
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compatibility" / "client-sdk"))
sys.path.insert(0, str(ROOT))


class RuntimeCompatibility(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {
            "GENRE_PACKS_PATH": str(ROOT / "genre_packs.json"),
            "WORKFLOW_PATH": str(ROOT / "workflow" / "wakeup_api.json"),
            "LATEST_DIR": str(self.root / "latest"),
            "ARCHIVE_DIR": str(self.root / "archive"),
            "GPU_RUNNER_CLIENT_STATE": str(self.root / "state"),
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.module = importlib.reload(importlib.import_module("app"))
        self.client = self.module.app.test_client()
        self.network = patch("requests.sessions.Session.request", side_effect=AssertionError("offline test forbids network"))
        self.network.start()
        self.addCleanup(self.network.stop)
        Path(self.module.LATEST_DIR).mkdir()

    def test_health_and_invalid_identity_never_start_backend_or_thread(self):
        with patch.object(self.module.threading, "Thread", side_effect=AssertionError("no worker expected")):
            self.assertEqual(self.client.get("/health").json, {"ok": True})
            response = self.client.post("/swap_and_generate", headers={"X-Request-ID": "../invalid"})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json, {"error": "Invalid X-Request-ID"})

    def test_busy_slot_preserves_audio_and_cannot_launch_second_worker(self):
        latest = Path(self.module.LATEST_DIR) / self.module.LATEST_NAME
        latest.write_bytes(b"existing audio fixture")
        operation = self.module._output_slot().begin("swap-and-generate", "fixture-held")
        try:
            with patch.object(self.module.threading, "Thread", side_effect=AssertionError("no second worker")):
                response = self.client.post("/swap_and_generate", headers={"X-Request-ID": "fixture-second"})
            self.assertEqual(response.json["status"], "busy")
            self.assertEqual(response.json["job_id"], "fixture-held")
            self.assertEqual(latest.read_bytes(), b"existing audio fixture")
            self.assertFalse(Path(self.module.ARCHIVE_DIR).exists())
        finally:
            operation.finish(error="fixture complete")

    def test_known_generation_failure_preserves_promoted_audio_and_failed_status(self):
        from aihub_gpu_runner.client import KnownJobFailure
        latest = Path(self.module.LATEST_DIR) / self.module.LATEST_NAME
        latest.write_bytes(b"already promoted fixture")
        operation = self.module._output_slot().begin("swap-and-generate", "fixture-failure")
        operation.record["response"] = {"job_id": "fixture-failure", "promoted": True, "status": "started"}
        self.module._set_job(operation, status="queued", promoted=True)
        with patch.object(self.module, "ollama_make_tags_and_lyrics", side_effect=KnownJobFailure("fixture retired without replay")), \
             patch.object(self.module, "comfy_generate_mp3", side_effect=AssertionError("no inference after failure")):
            self.module.generate_new("fixture-failure", operation)
        status = self.module._get_job("fixture-failure")
        self.assertEqual(status["status"], "error")
        self.assertTrue(status["promoted"])
        self.assertEqual(latest.read_bytes(), b"already promoted fixture")
        self.assertFalse((Path(self.module.LATEST_DIR) / self.module.NEW_NAME).exists())
        self.assertIsNone(self.module.CURRENT_OPERATION.get())
        self.assertEqual(self.module._output_slot().operation_status("fixture-failure")["status"], "error")


if __name__ == "__main__":
    unittest.main()
