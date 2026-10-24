# Arjun voice agent

A hands-free desktop voice assistant for an [OpenClaw](https://github.com/openclaw/openclaw) agent on Linux.
Say **"Hey Arjun"** and talk: a Gemini Live voice model handles the conversation in real time and hands
real work to your OpenClaw agent, which keeps running while you talk, interrupt or ask how it's going.

- **Wake phrase:** "Hey Arjun" (also "Hi/OK Arjun"), detected locally with Silero VAD + Whisper
- **Live conversation:** full-duplex Gemini Live call over the OpenClaw Gateway relay; interrupt by talking
- **Delegation:** tasks go to the OpenClaw agent and survive interruptions; "stop"/"cancel" aborts them
- **One continuous chat** until you say "new chat"
- **On-screen wave** (bottom of the screen) with play/stop and mute buttons, plus a **tray icon** menu

## How it works

```
mic ─► Silero VAD + Whisper ("hey arjun") ─► OpenClaw Gateway (talk.session, gateway-relay)
                                                 ├─ Gemini Live: listens and speaks
                                                 └─ your OpenClaw agent: does the actual tasks
```

| File | Role |
|---|---|
| `arjun_app.py` | Tray app: shows the overlay and runs the voice engine as a supervised child process |
| `arjun_live.py` | Voice engine: wake phrase, live call, task delegation, mute/play/stop/new-chat controls |
| `overlay.py` | Wave overlay with play/stop and mute buttons (Wayland layer-shell, click-through) |
| `arjun_voice.py` | Older turn-based pipeline (openWakeWord → Whisper → OpenClaw → Kokoro), kept as a fallback |
| `deploy/` | systemd user unit, env file template, PipeWire echo-cancel config |

## Requirements

- Linux on **Wayland** with a compositor that supports layer-shell and a StatusNotifier tray
  (tested on Pop!_OS 24.04 with COSMIC), and **PipeWire**
- An **NVIDIA GPU** for Whisper (`large-v3-turbo` uses about 1.5 GB of VRAM)
- An **OpenClaw Gateway** running locally with:
  - the Google realtime voice provider configured (a Gemini API key), so `talk.catalog` reports realtime as ready
  - an agent model for the delegated tasks (the default is Sonnet 5.5 through the Anthropic API)
- Python 3.11 for the engine (via [uv](https://github.com/astral-sh/uv)) and the system `python3` for the tray app

## Setup

### 1. System packages

```bash
sudo apt install libportaudio2 espeak-ng \
  python3-gi python3-gi-cairo gir1.2-gtk-3.0 gir1.2-gtklayershell-0.1 gir1.2-ayatanaappindicator3-0.1
```

`espeak-ng` is only needed by the older `arjun_voice.py` pipeline (Kokoro).

### 2. Python environment

```bash
git clone https://github.com/karthik24iyer/arjun-voice-agent.git ~/Documents/Projects/arjun-voice
cd ~/Documents/Projects/arjun-voice
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Whisper and Silero download their models on first start.

### 3. Echo-cancelled microphone

With speakers, the assistant would otherwise hear itself. Find your mic's name, put it in the config and
restart PipeWire:

```bash
pactl list short sources                      # e.g. alsa_input.usb-..._WEBCAM_...mono-fallback
mkdir -p ~/.config/pipewire/pipewire.conf.d
cp deploy/60-arjun-echo-cancel.conf ~/.config/pipewire/pipewire.conf.d/
sed -i 's/YOUR_MIC_SOURCE_NAME/<your mic source name>/' ~/.config/pipewire/pipewire.conf.d/60-arjun-echo-cancel.conf
systemctl --user restart pipewire pipewire-pulse wireplumber
pactl set-default-source arjun_ec_source      # optional, keeps streams on the cleaned mic
```

### 4. Local settings

```bash
mkdir -p ~/.config/arjun
cp deploy/arjun.env.example ~/.config/arjun/arjun.env
# set ARJUN_RAW_MIC to the same mic source name (used directly when you wear headphones)
```

### 5. OpenClaw realtime settings (recommended)

```bash
openclaw config set talk.realtime.silenceDurationMs 1500   # don't cut you off at short pauses
openclaw config set talk.realtime.instructions "Your name is Arjun. Before every openclaw_agent_consult call, first say one short natural filler such as 'Let me check that' or 'One moment', then call the tool. Tasks keep running in the background even if the person interrupts you: after any interruption, if a task was in progress, call openclaw_agent_consult again with the same question to collect its result. Keep spoken replies brief and conversational."
```

### 6. Install the service

```bash
cp deploy/arjun.service ~/.config/systemd/user/   # edit WorkingDirectory if you cloned elsewhere
systemctl --user daemon-reload
systemctl --user enable --now arjun
```

It starts with your desktop session; the mic icon appears in the panel and the wave at the bottom of the screen.

## Usage

- **"Hey Arjun, …"** opens a call; keep talking normally, follow-ups need no wake phrase.
  The call closes after 20 s of quiet or when you say "bye".
- **Interrupt** by talking (headphones), or by saying "Hey Arjun" (speakers, see below).
- **"New chat"** (or "start over") switches to a fresh OpenClaw session; otherwise the same chat continues.
  The current chat is kept in `~/.local/state/arjun-voice/live-session.json`.
- **Overlay:** ▶/■ starts or ends a call, the mic button mutes. Wave colours: grey idle, blue listening,
  purple working, green speaking, gold new chat; flat when muted or stopped.
- **Tray menu:** status, talk now / end call, mute, new chat, show wave, start at login,
  restart / stop / start Arjun, view logs, quit.

### Speakers vs headphones

The default output is checked every few seconds:

- **Headphones** (`device.form_factor` headphone/headset): the raw mic is used and you can talk over Arjun.
- **Speakers:** the echo-cancelled mic is used and, because Bluetooth delay defeats echo cancellation, the mic is
  muted towards the voice model while Arjun speaks (`ARJUN_DUPLEX=half`). Say "Hey Arjun" to interrupt him.
  With wired speakers you can try `ARJUN_DUPLEX=full`.

## Configuration

Set in `~/.config/arjun/arjun.env` (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `ARJUN_RAW_MIC` | `ARJUN_MIC` | Physical mic source, used with headphones |
| `ARJUN_MIC` | `arjun_ec_source` | Echo-cancelled mic, used with speakers |
| `ARJUN_MIC_GAIN` | `3.0` | Software gain for the echo-cancelled mic |
| `ARJUN_WAKE_NAME` | `arjun` | Name in the wake phrase ("hey <name>") |
| `ARJUN_STT_MODEL` | `large-v3-turbo` | faster-whisper model for wake detection |
| `ARJUN_DUPLEX` | `half` | `half` or `full` when using speakers |
| `ARJUN_LIVE_IDLE_SECONDS` | `20` | Hang up after this much quiet |
| `ARJUN_LIVE_PROVIDER` | `google` | OpenClaw realtime provider |
| `ARJUN_VOICE_MODEL` | `anthropic/claude-sonnet-5-5 --runtime openclaw` | Model pinned to the voice chat for delegated tasks |
| `OPENCLAW_GATEWAY_WS` | `ws://127.0.0.1:18789` | Gateway WebSocket |
| `OPENCLAW_GATEWAY_TOKEN` | from `~/.openclaw/openclaw.json` | Gateway token |

## Troubleshooting

```bash
systemctl --user status arjun
journalctl --user -u arjun -f          # what it heard, calls, tasks
```

- **Wake phrase missed:** check the log for `wake phrase heard`; speak a little closer to the mic.
- **It answers itself:** you're on speakers with `ARJUN_DUPLEX=full`, or the echo-cancel config isn't loaded
  (`pactl list short sources | grep arjun_ec_source`).
- **No voice / call fails to open:** `openclaw` must report realtime as ready for the Google provider.
