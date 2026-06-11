from flask import Flask, jsonify
from urllib.parse import urlencode
import os, shutil, tempfile, datetime as dt, threading, time
import json, hashlib, random
import requests
import copy
import traceback
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
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    archive_path = os.path.join(ARCHIVE_DIR, f"wakeup-{stamp}.mp3")
    shutil.copyfile(path, archive_path)
    return archive_path

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

    # Use Ollama /api/chat for better control
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
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
        },
        timeout=OLLAMA_TIMEOUT
    )
    r.raise_for_status()
    resp = r.json()
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

def _find_file_records(obj):
    """
    Recursively find dicts that look like Comfy file records:
    {"filename": "...", "subfolder": "...", "type": "..."}
    """
    found = []
    if isinstance(obj, dict):
        if "filename" in obj and isinstance(obj["filename"], str):
            found.append({
                "filename": obj["filename"],
                "subfolder": obj.get("subfolder", "") or "",
                "type": obj.get("type", "output") or "output",
            })
        for v in obj.values():
            found.extend(_find_file_records(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_find_file_records(item))
    return found

def comfy_generate_mp3(job_id: str, workflow: dict) -> str:
    client_id = f"wake-{random.randint(100000,999999)}"
    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30
    )

    if not r.ok:
        raise RuntimeError(f"ComfyUI /prompt failed: HTTP {r.status_code}: {r.text[:3000]}")

    prompt_id = r.json()["prompt_id"]
    JOBS[job_id]["prompt_id"] = prompt_id

    deadline = time.time() + 1800  # 30 min max
    while time.time() < deadline:
        h = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        h.raise_for_status()
        hist = h.json()
        item = hist.get(prompt_id)

        if item:
            # Search everything Comfy returned for file records
            file_recs = _find_file_records(item)

            # Prefer mp3
            mp3s = [f for f in file_recs if f["filename"].lower().endswith(".mp3")]
            pick = (mp3s or file_recs)
            if pick:
                f0 = pick[0]
                qs = urlencode({"filename": f0["filename"], "subfolder": f0["subfolder"], "type": f0["type"]})
                url = f"{COMFY_URL}/view?{qs}"
                JOBS[job_id]["mp3_url"] = url
                return url

        time.sleep(3)

    raise RuntimeError("Timed out waiting for ComfyUI audio output")

def generate_new(job_id: str):
    try:
        JOBS[job_id]["status"] = "running"

        pack = pick_genre_pack()
        JOBS[job_id]["pack"] = pack["name"]

        gendered_tags, vocal_gender = apply_vocal_gender_tags(pack["tags"])
        JOBS[job_id]["vocal_gender"] = vocal_gender
        JOBS[job_id]["base_tags_preview"] = gendered_tags[:300]

        # 1) Ask Ollama for tags+lyrics
        tl = ollama_make_tags_and_lyrics(pack["name"], gendered_tags)
        tags = tl["tags"]
        lyrics = tl["lyrics"]
        JOBS[job_id]["tags_preview"] = tags[:300]
        JOBS[job_id]["lyrics_preview"] = lyrics[:300]

        # 2) Build workflow from your API JSON (copy your dict here or load it)
        workflow = copy.deepcopy(WORKFLOW_TEMPLATE)

        # ACE-Step 1.5 workflow source nodes
        # 225 feeds TextEncodeAceStepAudio1.5 tags / genre prompt
        # 226 feeds TextEncodeAceStepAudio1.5 lyrics / song prompt
        # 117 feeds both duration and EmptyAceStep1.5LatentAudio seconds
        # 409 feeds the rgthree Seed node
        workflow["225"]["inputs"]["value"] = tags
        workflow["226"]["inputs"]["value"] = lyrics
        workflow["117"]["inputs"]["value"] = TARGET_SECONDS
        workflow["409"]["inputs"]["seed"] = random.randint(1, 999_999_999_999_999)

        # 3) Run Comfy and download MP3
        mp3_url = comfy_generate_mp3(job_id, workflow)

        dl = requests.get(mp3_url, timeout=300)
        dl.raise_for_status()

        # 4) Write to wakeup-new.mp3 atomically + chmod 644
        new_path = os.path.join(LATEST_DIR, NEW_NAME)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(new_path))
        os.close(fd)
        with open(tmp, "wb") as f:
            f.write(dl.content)
        os.replace(tmp, new_path)
        os.chmod(new_path, 0o644)

        JOBS[job_id]["status"] = "done"

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["traceback"] = traceback.format_exc()

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/swap_and_generate")
def swap_and_generate():
    latest_path = os.path.join(LATEST_DIR, LATEST_NAME)
    new_path    = os.path.join(LATEST_DIR, NEW_NAME)

    archived = archive_file(latest_path)

    promoted = False
    if os.path.exists(new_path):
        # Promote NEW -> LATEST atomically
        os.replace(new_path, latest_path)
        os.chmod(latest_path, 0o644)
        promoted = True

    job_id = dt.datetime.now().strftime("%Y%m%d%H%M%S%f")
    JOBS[job_id] = {"status": "queued", "archived": archived, "promoted": promoted}
    t = threading.Thread(target=generate_new, args=(job_id,), daemon=True)
    t.start()

    return jsonify({
        "job_id": job_id,
        "archived": archived,
        "promoted": promoted,
        "latest": f"{LATEST_DIR}/{LATEST_NAME}",
        "new": f"{LATEST_DIR}/{NEW_NAME}",
        "status": "started"
    })

@app.get("/debug/comfy_history/<job_id>")
def debug_comfy_history(job_id):
    j = JOBS.get(job_id, {})
    pid = j.get("prompt_id")
    if not pid:
        return jsonify({"error": "no prompt_id yet", "job": j}), 400
    h = requests.get(f"{COMFY_URL}/history/{pid}", timeout=30)
    return jsonify(h.json())

@app.get("/job/<job_id>")
def job_status(job_id):
    return jsonify(JOBS.get(job_id, {"status": "unknown"}))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8788)