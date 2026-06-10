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
```

Example .env:

```code
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

```code
rest_command:
  wake_swap_and_generate:
    url: "http://YOUR-SERVICE-HOST:8788/swap_and_generate"
    method: POST
    timeout: 30
```

Your media player automation can play:

```text
wakeup-latest.mp3
```

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

## Genre Packs

Genre/style prompts live in:

```text
genre_packs.json
```

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

```text
This project expects an exported ComfyUI API workflow (sample included, but may require additional nodes), not the visual/editor workflow.
```

The default ACE-Step 1.5 node mapping used by app.py is:

```text
225 = genre/style prompt source
226 = lyrics prompt source
117 = duration source
409 = seed source
```

If you export a different workflow, these node IDs may need to be updated in app.py.

## Future Enhancements
Turn into a Home Assistant adaptive soundtrack engine, generating generate context-aware music for household events.

Welcome home / goodbye songs

```text
Arrive home → short triumphant welcome-home song
Leave for work → upbeat “go crush the day” song
Arriving home late → softer low-energy welcome song
```
Example triggers

```text
person.HA_user changed to home
phone connected to Wi-Fi
garage/door opened
WireGuard disabled because you’re home
```

Task-completion victory songs

```Text
Dishwasher unloaded → tiny heroic fanfare
Laundry moved to dryer → goofy achievement song
Office cleaned → orchestral victory theme
Garage tidied → dad-rock “job well done” anthem
Workout completed → power metal reward song
```

Example triggers

```text
camera confirms room state changed
door/contact sensors
smart plug usage patterns
manual dashboard button
voice command: “JARVIS, I finished cleaning”
```

Cleaning music with camera/person detection

Possible logic

```text
Person detected in kitchen for >5 minutes
AND vacuum is not running
AND time is daytime/evening
→ start cleaning playlist or generate cleaning song
```
```text
Camera sees clutter level high
Person starts moving around kitchen
→ generate “cleaning montage” music
```
```text
Camera detects likely cleaning activity
→ HA asks/announces: “Cleaning mode, sir/madam?”
→ user confirms by voice/button
→ music starts
```

Dynamic “room soundtrack” mode
```text
Kitchen morning → jazz soul sunrise
Office focus → low-lyric cinematic synthwave
Garage work → southern rock / blues rock
Living room evening → cozy acoustic / jazz
Bathroom shower → ridiculous power ballad
```

Example triggers based on occupancy and environmental conditions

```text
room = kitchen
time = morning
weather = rainy
presence = user
mode = normal
```

Weather-aware songs

```text
Snowy morning → cozy orchestral folk wake-up song
Rainy morning → dark cinematic synthwave
Sunny weekend → funk/disco morning
Storm warning → dramatic sea shanty
Heat warning → lazy desert blues
```

Prompt could include weather text

```code
{
  "weather": "rainy, 8C, dark morning",
  "event": "wake_up",
  "energy": "medium"
}
```

Calendar-aware songs
See AI based calendar at:
```http
https://github.com/bojinda/dashboard-generator
```
Calendar pulls events from Google and Home Assistant entities

```text
Early shift → aggressive high-energy wake-up track
Day off → relaxed jazz/soul song
Union meeting day → heroic speech-like anthem
Travel day → road song
Appointment day → gentle but punctual reminder song
```

```
Today includes: work, union meeting, drive to Toronto
Generate a confident morning song about getting moving.
```

Personalized recurring characters

```text
Captain Coffee
The Morning Goblin
Sir Laundry of the Dryer
JARVIS as narrator
A fake band name that changes by genre
```

Store as small SQLite file
```text
last_10_songs
favorite_genres
running_jokes
recent_events
```

House event jingles

```text
Doorbell → custom 8-second chime
Washer done → laundry jingle
Dryer done → dryer victory riff
Dinner timer → medieval feast horn
Trash night → ominous garbage anthem
Server alert → dramatic warning sting
```
Usage
```text
duration_seconds = 8–20
lyrics = none or one phrase
```

Server / homelab status

```text
Proxmox rebooted successfully → triumphant orchestral fanfare
Backup completed → calm success jingle
Disk space warning → ominous synth alert
GPU job finished → sci-fi lab success music
ComfyUI crashed → sad trombone, obviously
```

Possible triggers
```text
Uptime Kuma
Home Assistant sensors
Docker health checks
Proxmox API
```

Voice-command generated songs via Home Assistant
```text
“Generate me a two-minute song about cleaning the kitchen.”
“Make a goodbye song for work.”
“Make a dramatic song because the laundry is finally done.”
“Generate battle music for fixing the server.”
```

Possible payload
```code
{
  "topic": "I finally fixed Docker DNS",
  "style": "heroic power metal",
  "duration_seconds": 90
}
```