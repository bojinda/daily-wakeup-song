# Daily Wake-Up Song

Remember clock radios? They were great for waking up to a new song every day! This project brings back that experience by generating a new daily wake-up song for you!

Automated daily wake-up song generation using **Ollama**, **ComfyUI**, and **ACE-Step 1.5**.

This project runs a small Flask service that:

1. Selects a genre/style pack
2. Asks an Ollama model to generate an ACE-Step style prompt and lyrics
3. Injects those values into a ComfyUI ACE-Step 1.5 API workflow
4. Generates a new MP3 wake-up song
5. Promotes the previous generated song to the “latest” wake-up song on the next run
6. Archives older wake-up songs automatically

It is designed to work well with Home Assistant REST commands, cron jobs, or any automation system that can send an HTTP POST request.

## How it works

The service exposes:

```http
POST /swap_and_generate
```

When called, it:

Archives the current wakeup-latest.mp3
Promotes wakeup-new.mp3 to wakeup-latest.mp3, if a new song exists
Starts generating the next wakeup-new.mp3 in the background

This means your automation can play the latest already-generated song immediately, while the next song is prepared for the following run.

Requirements
```text
Docker and Docker Compose
A running ComfyUI instance
A working ACE-Step 1.5 ComfyUI API workflow
A running Ollama server
At least one Ollama text model installed
Storage folders for latest and archived wake-up songs
```
Project Files
```text
app.py                   Flask service
docker-compose.yml       Docker Compose service definition
Dockerfile               Python container definition
genre_packs.json         Genre/style prompt packs
.env.example             Example environment configuration
workflow/wakeup_api.json ComfyUI API workflow
```

## Setup
Clone the repository:
```code
git clone https://github.com/YOUR-USERNAME/daily-wakeup-song.git
cd daily-wakeup-song
```
Create your environment file:
```code
cp .env.example .env
nano .env

Example .env:

TZ=America/Toronto

COMFY_URL=http://YOUR-COMFYUI-HOST:8188
OLLAMA_URL=http://YOUR-OLLAMA-HOST:11434
OLLAMA_MODEL=qwen2.5:32b
OLLAMA_TIMEOUT=300

WORKFLOW_PATH=/app/workflow/wakeup_api.json
TARGET_SECONDS=135

GENRE_PACKS_PATH=/app/genre_packs.json
GENRE_PICK_MODE=random

LATEST_DIR=/latest-wake
ARCHIVE_DIR=/archive-wake
SEED_MP3=/seed/test.mp3

LATEST_NAME=wakeup-latest.mp3
NEW_NAME=wakeup-new.mp3
```
Start the service:

```code
docker compose up -d --build
```

Check logs:

```code
docker logs -f wake-songservice
```

## Home Assistant Example

Example REST command:

``code
rest_command:
  wake_swap_and_generate:
    url: "http://YOUR-SERVICE-HOST:8788/swap_and_generate"
    method: POST
    timeout: 30
```

Your media player automation can play:

wakeup-latest.mp3

from wherever you expose the latest wake-up song file.

```text
API Endpoints
Health Check
GET /health
```

Returns:

```text
{"ok": true}
Start Wake-Up Song Rotation
POST /swap_and_generate
```

Returns a job ID and starts generation in the background.

```code
Check Job Status
GET /job/<job_id>
```

Example response:

```code
{
  "status": "done",
  "pack": "cinematic drum and bass",
  "prompt_id": "example-prompt-id",
  "mp3_url": "http://comfyui:8188/view?filename=AceStep_00009.mp3&subfolder=audio&type=output"
}
```

Debug ComfyUI History

```code
GET /debug/comfy_history/<job_id>
```

Useful for troubleshooting ComfyUI workflow failures.

Genre Packs

Genre/style prompts live in:

genre_packs.json

Each pack needs:
```code
{
  "name": "cinematic drum and bass",
  "tags": "A rich natural-language production prompt..."
}
```

The app can select packs randomly or deterministically by day.

```code
GENRE_PICK_MODE=random
```

or:

```code
GENRE_PICK_MODE=daily
```

ComfyUI Workflow Notes

This project expects an exported ComfyUI API workflow (sample included, but may require additional nodes), not the visual/editor workflow.

The default ACE-Step 1.5 node mapping used by app.py is:

```text
225 = genre/style prompt source
226 = lyrics prompt source
117 = duration source
409 = seed source
```

If you export a different workflow, these node IDs may need to be updated in app.py.