from flask import Flask, jsonify, request
from urllib.parse import urlencode
import os, shutil, tempfile, datetime as dt, threading, time
import json, hashlib, random
import requests
import copy
import traceback
from gpu_runner import chat as runner_chat, music as runner_music
from aihub_gpu_runner.client import CURRENT_OPERATION, OutputSlot, OutputBusy, CoordinationError, publish_bytes
from aihub_gpu_runner.core import check_id
os.umask(0o022)

OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))
GENRE_PACKS_PATH = os.environ.get("GENRE_PACKS_PATH", "/app/genre_packs.json")
LATEST_DIR  = os.environ.get("LATEST_DIR", "/latest-wake")
ARCHIVE_DIR = os.environ.get("ARCHIVE_DIR", "/archive-wake")
SEED_MP3    = os.environ.get("SEED_MP3", "/seed/test.mp3")

LATEST_NAME = os.environ.get("LATEST_NAME", "wakeup-latest.mp3")
NEW_NAME    = os.environ.get("NEW_NAME", "wakeup-new.mp3")

COMFY_URL  = os.environ.get("COMFY_URL", "http://comfyui:8188")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b")

TARGET_SECONDS = int(os.environ.get("TARGET_SECONDS", "120"))

app = Flask(__name__)
JOBS = {}

VOCAL_GENDER_PROFILES = [
    {
        "gender": "male",
        "lead_vocal": "a powerful male lead vocalist",
        "chorus_vocal": "a large male chorus",
        "voice_register": "deep, resonant baritone vocals",
    },
    {
        "gender": "female",
        "lead_vocal": "a powerful female lead vocalist",
        "chorus_vocal": "a large female chorus",
        "voice_register": "rich, expressive alto vocals",
    },
]

def apply_vocal_gender_tags(tags: str) -> tuple[str, str]:
    profile = random.choice(VOCAL_GENDER_PROFILES)

    try:
        tags = tags.format(
            lead_vocal=profile["lead_vocal"],
            chorus_vocal=profile["chorus_vocal"],
            voice_register=profile["voice_register"],
        )
    except KeyError as e:
        raise RuntimeError(f"Unknown vocal placeholder in genre tags: {e}")

    return tags, profile["gender"]

def load_genre_packs():
    with open(GENRE_PACKS_PATH, "r", encoding="utf-8") as f:
        packs = json.load(f)

    if not isinstance(packs, list) or not packs:
        raise RuntimeError(f"No genre packs found in {GENRE_PACKS_PATH}")

    for pack in packs:
        if "name" not in pack or "tags" not in pack:
            raise RuntimeError("Each genre pack must include name and tags")

    return packs

GENRE_PACKS = load_genre_packs()

WORKFLOW_PATH = os.environ.get("WORKFLOW_PATH", "/app/workflow/wakeup_api.json")

with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
    WORKFLOW_TEMPLATE = json.load(f)

def atomic_copy(src_path: str, dst_path: str):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(dst_path))
    os.close(fd)
    shutil.copyfile(src_path, tmp)
    os.replace(tmp, dst_path)      # atomic replace
    os.chmod(dst_path, 0o644)      # <-- ensure nginx can read

def archive_file(path: str):
    if not os.path.exists(path):
        return None
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    moment = dt.datetime.now()
    while True:
        stamp = moment.strftime("%Y-%m-%d_%H%M%S")
        archive_path = os.path.join(ARCHIVE_DIR, f"wakeup-{stamp}.mp3")
        try:
            with open(archive_path, "xb") as destination, open(path, "rb") as source:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            return archive_path
        except FileExistsError:
            moment += dt.timedelta(seconds=1)
        except BaseException:
            try:
                os.unlink(archive_path)
            except FileNotFoundError:
                pass
            raise


def pick_genre_pack():
    mode = os.environ.get("GENRE_PICK_MODE", "random").lower()

    if mode == "daily":
        day_key = dt.date.today().isoformat().encode()
        idx = int(hashlib.sha256(day_key).hexdigest(), 16) % len(GENRE_PACKS)
        return GENRE_PACKS[idx]

    return random.choice(GENRE_PACKS)

def ollama_make_tags_and_lyrics(pack_name: str, pack_tags: str) -> dict:
    system = (
        "You generate music prompts and lyrics for ComfyUI ACE-Step 1.5.\n"
        "Return ONLY valid JSON with exactly two keys: tags, lyrics.\n"
        "tags: one rich natural-language production prompt, 45-90 words, describing genre, vocalist, instrumentation, groove, mood, energy, tempo, and mix style.\n"
        "lyrics: multi-line English lyrics using [intro-short], [verse], [chorus], [bridge], [outro-short].\n"
        "Rules:\n"
        "- English only.\n"
        "- Make it a 'good morning' wake-up song.\n"
        "- Target about 2 minutes.\n"
        "- Include a catchy chorus and repeat it.\n"
        "- Do not use backslashes.\n"
        "- Do NOT output any other keys or commentary.\n"
    )

    user = {
        "genre_pack": pack_name,
        "base_tags": pack_tags,
        "language": "en_only",
        "duration_seconds": TARGET_SECONDS,
        "seed_phrase": "good morning"
    }

    resp = runner_chat({
            "model": OLLAMA_MODEL,
            "messages": [
                {"role":"system","content":system},
                {"role":"user","content":json.dumps(user)}
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0.8,
                "num_predict": 900
            }    
        }, on_status=_runner_progress)
    content = (resp.get("message", {}).get("content") or "").strip()

    out = {}
    if content:
        try:
            out = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = content[start:end + 1]

                # Common LLM mistake: JSON strings containing \' escape.
                candidate = candidate.replace("\\'", "'")

                try:
                    out = json.loads(candidate)
                except json.JSONDecodeError:
                    out = {}

    if not isinstance(out, dict):
        out = {}

    if not out.get("tags"):
        out["tags"] = pack_tags

    if not out.get("lyrics"):
        out["lyrics"] = (
            "[intro-short]\n"
            "Good morning, the day is calling\n\n"
            "[verse]\n"
            "The sun is rising through the window\n"
            "The night is fading from the sky\n"
            "I take a breath and find my footing\n"
            "Today is mine and I know why\n\n"
            "[chorus]\n"
            "Good morning, rise up into the light\n"
            "Good morning, leave behind the night\n"
            "Good morning, let the whole world see\n"
            "The best of what today can be\n\n"
            "[bridge]\n"
            "One step forward, one breath stronger\n"
            "I can carry what I need\n\n"
            "[chorus]\n"
            "Good morning, rise up into the light\n"
            "Good morning, leave behind the night\n"
            "Good morning, let the whole world see\n"
            "The best of what today can be\n\n"
            "[outro-short]\n"
            "Good morning"
        )

    out["tags"] = out["tags"].strip()
    out["lyrics"] = out["lyrics"].strip()
    return out

_OUTPUT_SLOT = None

def _output_slot():
    global _OUTPUT_SLOT
    if _OUTPUT_SLOT is None:
        _OUTPUT_SLOT = OutputSlot(os.environ.get("GPU_RUNNER_CLIENT_STATE", "/var/lib/wake-song-gpu-client"),
                                  [os.path.join(LATEST_DIR, LATEST_NAME), os.path.join(LATEST_DIR, NEW_NAME), ARCHIVE_DIR])
    return _OUTPUT_SLOT

def _set_job(operation, **fields):
    operation.update_details(**fields)
    JOBS[operation.request_id] = dict(operation.record.get("details", {}))

def _get_job(job_id):
    try:
        check_id(job_id)
    except ValueError:
        return {"status": "unknown"}
    record = _output_slot().operation_status(job_id)
    if record is None:
        return {"status": "unknown"}
    job = dict(record.get("details", {}))
    if record["status"] == "error" and job.get("status") in ("queued", "running"):
        job.update(status="error", error="interrupted_output_operation_no_replay")
    return job

def _runner_progress(status):
    operation = CURRENT_OPERATION.get()
    if operation is None:
        raise CoordinationError("owned_music_operation_required")
    fields = {"runner_job_id": status["request_id"], "runner_state": status["state"]}
    prompt_id = status.get("backend_detail", {}).get("prompt_id")
    if prompt_id:
        fields["prompt_id"] = prompt_id
    _set_job(operation, **fields)

def comfy_generate_mp3(job_id: str, workflow: dict):
    data, metadata = runner_music(workflow, on_status=_runner_progress)
    url = COMFY_URL.rstrip("/") + "/view?" + urlencode({
        "filename": metadata["filename"], "subfolder": metadata.get("subfolder", ""), "type": "output"})
    _set_job(CURRENT_OPERATION.get(), prompt_id=metadata["prompt_id"], mp3_url=url)
    return data

def generate_new(job_id: str, operation):
    token = CURRENT_OPERATION.set(operation)
    try:
        _set_job(operation, status="running")
        pack = operation.snapshot("genre-choice", pick_genre_pack())
        gendered_tags, vocal_gender = apply_vocal_gender_tags(pack["tags"])
        voice = operation.snapshot("vocal-choice", {"tags": gendered_tags, "gender": vocal_gender})
        _set_job(operation, pack=pack["name"], vocal_gender=voice["gender"], base_tags_preview=voice["tags"][:300])

        # GPU1 request and owned cleanup/release complete before GPU0 admission.
        tl = ollama_make_tags_and_lyrics(pack["name"], voice["tags"])
        tags, lyrics = tl["tags"], tl["lyrics"]
        _set_job(operation, tags_preview=tags[:300], lyrics_preview=lyrics[:300])

        workflow = copy.deepcopy(WORKFLOW_TEMPLATE)
        workflow["225"]["inputs"]["value"] = tags
        workflow["226"]["inputs"]["value"] = lyrics
        workflow["117"]["inputs"]["value"] = TARGET_SECONDS
        workflow["409"]["inputs"]["seed"] = random.randint(1, 999_999_999_999_999)

        audio = comfy_generate_mp3(job_id, workflow)
        publish_bytes(os.path.join(LATEST_DIR, NEW_NAME), audio, mode=0o644)
        _set_job(operation, status="done")
        operation.finish(operation.record["response"])
    except Exception as exc:
        if operation.lock is not None:
            try:
                _set_job(operation, status="error", error=str(exc), traceback=traceback.format_exc())
            finally:
                operation.finish(error=str(exc))
        else:
            # Publication may be complete but metadata synchronization failed.
            # The CPU guard is released: never overwrite a subsequent owner.
            print(f"Music status synchronization failed for {job_id}: {exc}", flush=True)
    finally:
        CURRENT_OPERATION.reset(token)

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/swap_and_generate")
def swap_and_generate():
    job_id = request.headers.get("X-Request-ID") or dt.datetime.now().strftime("%Y%m%d%H%M%S%f")
    try:
        operation = _output_slot().begin("swap-and-generate", job_id)
    except OutputBusy:
        active = _output_slot().status()
        details = active.get("details", {})
        return jsonify({"job_id": active.get("request_id"), "archived": details.get("archived"),
                        "promoted": details.get("promoted", False), "latest": f"{LATEST_DIR}/{LATEST_NAME}",
                        "new": f"{LATEST_DIR}/{NEW_NAME}", "status": "busy"})
    except ValueError:
        return jsonify({"error": "Invalid X-Request-ID"}), 400
    except CoordinationError as exc:
        return jsonify({"job_id": job_id, "status": "error", "error": str(exc)}), 503
    if operation.cached:
        return jsonify(operation.record["response"])
    try:
        _set_job(operation, status="queued", archived=None, promoted=False)
        latest_path = os.path.join(LATEST_DIR, LATEST_NAME)
        new_path = os.path.join(LATEST_DIR, NEW_NAME)
        archived = archive_file(latest_path)
        promoted = False
        if os.path.exists(new_path):
            os.chmod(new_path, 0o644)
            os.replace(new_path, latest_path)
            promoted = True
        response = {"job_id": job_id, "archived": archived, "promoted": promoted,
                    "latest": f"{LATEST_DIR}/{LATEST_NAME}", "new": f"{LATEST_DIR}/{NEW_NAME}", "status": "started"}
        operation.record["response"] = response
        _set_job(operation, archived=archived, promoted=promoted)
        threading.Thread(target=generate_new, args=(job_id, operation), daemon=True).start()
        return jsonify(response)
    except Exception as exc:
        try:
            _set_job(operation, status="error", error=str(exc))
        finally:
            operation.finish(error=str(exc))
        return jsonify({"job_id": job_id, "status": "error", "error": str(exc)}), 500

@app.get("/debug/comfy_history/<job_id>")
def debug_comfy_history(job_id):
    j = _get_job(job_id)
    pid = j.get("prompt_id")
    if not pid:
        return jsonify({"error": "no prompt_id yet", "job": j}), 400
    h = requests.get(f"{COMFY_URL}/history/{pid}", timeout=30)
    return jsonify(h.json())

@app.get("/job/<job_id>")
def job_status(job_id):
    return jsonify(_get_job(job_id))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8788)
