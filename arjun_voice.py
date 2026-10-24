#!/usr/bin/env python3
"""Hey Arjun: local wake word -> Whisper STT -> OpenClaw agent -> Kokoro TTS.

The wake word stays live while Arjun thinks or speaks; saying it again interrupts him.
"""
import asyncio
import collections
import json
import logging
import os
import queue
import re
import socket
import threading
import time
import uuid
from pathlib import Path

MIC = os.environ.get("ARJUN_MIC", "arjun_ec_source")  # echo-cancelled mic (deploy/60-arjun-echo-cancel.conf)
os.environ["PULSE_SOURCE"] = MIC  # must be set before PortAudio opens the pulse device

import numpy as np
import sounddevice as sd
import torch
import websockets
from faster_whisper import WhisperModel
from kokoro import KPipeline
from openwakeword.model import Model as WakeModel
from openwakeword.utils import download_models
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from silero_vad import load_silero_vad

# --- Configuration ---
WAKEWORD = os.environ.get("ARJUN_WAKEWORD", "hey_jarvis")  # model name or path to a custom .onnx
WAKE_THRESHOLD = float(os.environ.get("ARJUN_WAKE_THRESHOLD", "0.4"))
MIC_GAIN = float(os.environ.get("ARJUN_MIC_GAIN", "3.0"))  # webcam mics are quiet at a distance
VOICE = os.environ.get("ARJUN_VOICE", "am_michael")
VOICE_SPEED = float(os.environ.get("ARJUN_VOICE_SPEED", "1.25"))
STT_MODEL = os.environ.get("ARJUN_STT_MODEL", "large-v3-turbo")
VOICE_MODEL = os.environ.get("ARJUN_VOICE_MODEL", "anthropic/claude-sonnet-5-5 --runtime openclaw")
NEW_CHAT_IDLE = float(os.environ.get("ARJUN_NEW_CHAT_IDLE_MINUTES", "120")) * 60
NEW_CHAT_WORDS = ("new chat", "start over", "new conversation", "fresh start")
STATE_FILE = Path.home() / ".local/state/arjun-voice/session.json"  # current session key + last activity
FOLLOWUP_SECONDS = float(os.environ.get("ARJUN_FOLLOWUP_SECONDS", "6"))
GATEWAY_WS = os.environ.get("OPENCLAW_GATEWAY_WS", "ws://127.0.0.1:18789")
GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN") or json.loads(
    (Path.home() / ".openclaw/openclaw.json").read_text())["gateway"]["auth"]["token"]
OVERLAY_ADDR = ("127.0.0.1", 47820)  # overlay.py
REPLY_TIMEOUT = 180
VOICE_PREFIX = ("[VOICE — spoken via 'Hey Arjun' on the PC. Reply in plain spoken English: "
                "no markdown, lists, emoji or URLs. Keep it brief unless asked for detail.] ")
EXIT_WORDS = ("bye", "goodbye", "stop", "that's all", "that is all", "nothing else")

RATE = 16000
CHUNK = 1280  # 80 ms, what openWakeWord expects
VAD_FRAME = 512  # 32 ms, what Silero expects at 16 kHz
SILENCE_END = 1.0
MAX_UTTERANCE = 20
WAKE_COOLDOWN = 1.5
ECHO_TAIL = 0.3  # ignore the room's echo of Arjun's voice when follow-up listening starts
PREROLL = 1.0  # audio kept from before the wake word fires, so "Hey Jarvis, what time is it" isn't cut
VAD_THRESHOLD = 0.4
_wake_words = Path(WAKEWORD).stem.split("_")
WAKE_PREFIX = re.compile(r"^\W*" + "".join(rf"(?:{w}\W*)?" for w in _wake_words[:-1])
                         + rf"{_wake_words[-1]}\b\W*", re.IGNORECASE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("arjun")

# --- Models ---
EspeakWrapper.set_library("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")  # bundled espeak ignores its data path
EspeakWrapper.set_data_path(None)
if not WAKEWORD.endswith(".onnx"):
    download_models([WAKEWORD])
wake = WakeModel(wakeword_models=[WAKEWORD], inference_framework="onnx")
vad = load_silero_vad()
stt = WhisperModel(STT_MODEL, device="cuda", compute_type="float16")
tts = KPipeline(lang_code=VOICE[0], repo_id="hexgrad/Kokoro-82M", device="cuda")

# --- Audio ---
mic_queue: "queue.Queue[np.ndarray]" = queue.Queue()


def on_audio(indata, frames, t, status):
    mic_queue.put(np.clip(indata[:, 0] * MIC_GAIN, -32768, 32767).astype(np.int16))


def flush_mic():
    while not mic_queue.empty():
        mic_queue.get_nowait()


_overlay_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def overlay(state: str = None, level: float = None):
    """Tell the on-screen wave what Arjun is doing; a missing overlay is ignored."""
    msg = {k: v for k, v in (("state", state), ("level", level)) if v is not None}
    try:
        _overlay_sock.sendto(json.dumps(msg).encode(), OVERLAY_ADDR)
    except OSError:
        pass


class Recorder:
    """Collects one utterance from mic chunks using Silero VAD."""

    def __init__(self, wait_seconds: float, skip_seconds: float = 0.0):
        vad.reset_states()
        self.frames, self.pending = [], np.zeros(0, dtype=np.float32)
        self.started, self.silence, self.heard, self.peak = False, 0.0, 0.0, 0.0
        self.wait, self.skip = wait_seconds, skip_seconds

    def feed(self, chunk: np.ndarray):
        """Return None while recording, the utterance when done, or an empty array on timeout."""
        self.pending = np.concatenate([self.pending, chunk.astype(np.float32) / 32768])
        while self.pending.size >= VAD_FRAME:
            frame, self.pending = self.pending[:VAD_FRAME], self.pending[VAD_FRAME:]
            self.heard += VAD_FRAME / RATE
            if self.heard < self.skip:
                continue
            p = vad(torch.from_numpy(frame), RATE).item()
            self.peak = max(self.peak, p)
            if p > VAD_THRESHOLD:
                self.started, self.silence = True, 0.0
            elif self.started:
                self.silence += VAD_FRAME / RATE
            if self.started:
                self.frames.append(frame)
        if not self.started:
            if self.heard > self.wait:
                log.info("no speech within %.0fs (peak speech prob %.2f)", self.wait, self.peak)
                return np.zeros(0, dtype=np.float32)
            return None
        if self.silence > SILENCE_END or len(self.frames) * VAD_FRAME / RATE > MAX_UTTERANCE:
            return np.concatenate(self.frames)
        return None


def transcribe(audio: np.ndarray) -> str:
    segments, _ = stt.transcribe(audio, language="en", vad_filter=False)
    return " ".join(s.text for s in segments).strip()


def clean_for_speech(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_#`~>|]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def speak(text: str, cancel: threading.Event):
    overlay("speaking")
    for result in tts(clean_for_speech(text), voice=VOICE, speed=VOICE_SPEED):
        if cancel.is_set():
            return
        audio = result.audio.numpy()
        sd.play(audio, 24000)
        start = time.time()
        while sd.get_stream().active:
            if cancel.is_set():
                sd.stop()
                return
            i = int((time.time() - start) * 24000)
            overlay(level=float(np.sqrt(np.mean(audio[i:i + 1200] ** 2))) * 4 if i < audio.size else 0.0)
            time.sleep(0.05)


# --- OpenClaw gateway ---
try:
    session = json.loads(STATE_FILE.read_text())
except (OSError, ValueError):
    session = {"key": "voice", "last": time.time()}


def save_session():
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(session))


async def ask_arjun(message: str, cancel: threading.Event, session_key: str):
    """Return Arjun's reply, or None if cancelled (the run is aborted on the gateway)."""
    deadline = time.time() + REPLY_TIMEOUT
    run_id = None
    async with websockets.connect(GATEWAY_WS, max_size=None) as ws:
        while time.time() < deadline:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), 0.2))
            except asyncio.TimeoutError:
                if cancel.is_set():
                    if run_id:
                        await ws.send(json.dumps({"type": "req", "id": "abort", "method": "chat.abort",
                                                  "params": {"sessionKey": session_key, "runId": run_id}}))
                        log.info("aborted run %s", run_id)
                    return None
                continue
            if msg.get("event") == "connect.challenge":
                await ws.send(json.dumps({"type": "req", "id": "connect", "method": "connect", "params": {
                    "minProtocol": 4, "maxProtocol": 4,
                    "client": {"id": "cli", "version": "1.0.0", "platform": "linux", "mode": "cli"},
                    "role": "operator", "scopes": ["operator.read", "operator.write"],
                    "auth": {"token": GATEWAY_TOKEN}}}))
            elif msg.get("id") == "connect":
                if not msg.get("ok"):
                    raise RuntimeError(f"gateway auth failed: {msg.get('error')}")
                await ws.send(json.dumps({"type": "req", "id": "chat", "method": "chat.send", "params": {
                    "sessionKey": session_key, "message": message, "deliver": False,
                    "timeoutMs": REPLY_TIMEOUT * 1000, "idempotencyKey": str(uuid.uuid4())}}))
            elif msg.get("id") == "chat":
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error", {}).get("message", "chat.send failed"))
                run_id = msg["payload"]["runId"]
            elif msg.get("event") == "chat" and run_id and msg["payload"].get("runId") == run_id:
                p = msg["payload"]
                if p.get("state") == "final":
                    return "".join(b.get("text", "") for b in p.get("message", {}).get("content", [])
                                   if b.get("type") == "text")
                if p.get("state") in ("error", "aborted"):
                    raise RuntimeError(p.get("errorMessage", p["state"]))
    raise RuntimeError("timed out waiting for Arjun")


def new_chat(cancel: threading.Event):
    """Move to a fresh OpenClaw session (a new key: /new needs admin scope) pinned to VOICE_MODEL."""
    overlay("new")
    t = time.time()
    key = "voice-" + time.strftime("%Y%m%d-%H%M%S")
    asyncio.run(ask_arjun(f"/model {VOICE_MODEL} -s", cancel, key))
    session["key"] = key
    log.info("new chat: %s", key)
    time.sleep(max(0.0, 1.0 - (time.time() - t)))  # keep the gold flash visible
    overlay("thinking")


class Turn(threading.Thread):
    """Transcribe, ask Arjun and speak the reply, stopping as soon as `cancel` is set."""

    def __init__(self, audio: np.ndarray):
        super().__init__(daemon=True)
        self.audio, self.cancel, self.follow_up, self.spoke = audio, threading.Event(), False, False

    def run(self):
        overlay("thinking")
        text = WAKE_PREFIX.sub("", transcribe(self.audio))
        log.info("heard: %s", text)
        if self.cancel.is_set():
            return
        if not text:  # only the wake word: keep listening for the question
            self.follow_up = True
            return
        if any(w in text.lower() for w in EXIT_WORDS) and len(text.split()) <= 4:
            speak("Bye.", self.cancel)
            return
        asked_new_chat = any(w in text.lower() for w in NEW_CHAT_WORDS) and len(text.split()) <= 5
        t = time.time()
        try:
            if asked_new_chat or t - session["last"] > NEW_CHAT_IDLE:
                new_chat(self.cancel)
            reply = ("Starting a new chat." if asked_new_chat else
                     asyncio.run(ask_arjun(VOICE_PREFIX + text, self.cancel, session["key"])))
        except Exception as e:
            log.error("agent failed: %s", e)
            reply = "Sorry, I couldn't get an answer for that."
        if reply is None:
            return
        session["last"] = time.time()
        save_session()
        log.info("reply in %.1fs: %s", time.time() - t, reply[:120])
        speak(reply, self.cancel)
        self.follow_up = self.spoke = not self.cancel.is_set()


def main():
    turn, recorder, last_wake, last_near = None, None, 0.0, 0.0
    history = collections.deque(maxlen=int(PREROLL * RATE / CHUNK))

    def listen(wait_seconds, skip_seconds=0.0, preroll=True):
        r = Recorder(wait_seconds, skip_seconds)
        for c in history if preroll else ():
            r.feed(c)
        return r

    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK, device="pulse",
                        callback=on_audio):
        log.info("listening for '%s' on %s", WAKEWORD, MIC)
        while True:
            chunk = mic_queue.get()
            history.append(chunk)
            score = max(wake.predict(chunk).values())
            if 0.2 <= score < WAKE_THRESHOLD and time.time() - max(last_wake, last_near) > WAKE_COOLDOWN:
                last_near = time.time()
                log.info("wake word near-miss (%.2f)", score)
            if score >= WAKE_THRESHOLD and time.time() - last_wake > WAKE_COOLDOWN:
                last_wake = time.time()
                if turn and turn.is_alive():
                    log.info("interrupted by wake word (%.2f)", score)
                    turn.cancel.set()
                    sd.stop()
                else:
                    log.info("wake word detected (%.2f)", score)
                turn = None
                overlay("listening")
                recorder = listen(wait_seconds=5)
                continue

            if recorder:
                overlay(level=float(np.sqrt(np.mean((chunk / 32768.0) ** 2))) * 6)
                audio = recorder.feed(chunk)
                if audio is not None:
                    recorder = None
                    if audio.size:
                        turn = Turn(audio)
                        turn.start()
                    else:
                        overlay("idle")
            elif turn and not turn.is_alive():
                overlay("listening" if turn.follow_up else "idle")
                if turn.follow_up:
                    log.info("listening for a follow-up")
                    if turn.spoke:  # don't let the tail of Arjun's own reply count as a question
                        flush_mic()
                        recorder = listen(FOLLOWUP_SECONDS, ECHO_TAIL, preroll=False)
                    else:
                        recorder = listen(FOLLOWUP_SECONDS)
                turn = None


if __name__ == "__main__":
    main()
