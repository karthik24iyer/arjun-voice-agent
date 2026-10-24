#!/usr/bin/env python3
"""Hey Arjun (live): "Hey Arjun" (Whisper) -> full-duplex Gemini Live session via the OpenClaw Gateway relay.

Gemini handles the conversation (barge-in, turn-taking, talking while work runs) and delegates real
tasks to the OpenClaw agent through `openclaw_agent_consult`; spoken corrections while a task runs go
through `openclaw_agent_control` (status / steer / cancel / follow-up).
"""
import asyncio
import base64
import collections
import difflib
import json
import logging
import os
import re
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import scipy.signal as ss
import sounddevice as sd
import torch
import websockets
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

# --- Configuration ---
# With speakers Arjun hears himself, so use the echo-cancelled mic; with headphones there is no echo, and the
# canceller only muffles the mic while the headphones play something, so use the raw mic.
MIC = os.environ.get("ARJUN_MIC", "arjun_ec_source")  # echo-cancelled mic (deploy/60-arjun-echo-cancel.conf)
RAW_MIC = os.environ.get("ARJUN_RAW_MIC") or MIC  # the physical mic (`pactl list short sources`), for headphones
WAKE_NAME = os.environ.get("ARJUN_WAKE_NAME", "arjun")  # wake phrase is "hey/hi/ok <name>"
STT_MODEL = os.environ.get("ARJUN_STT_MODEL", "large-v3-turbo")
MIC_GAIN = float(os.environ.get("ARJUN_MIC_GAIN", "3.0"))  # echo-cancelled mic only; the raw mic clips with gain
STATE_FILE = Path.home() / ".local/state/arjun-voice/live-session.json"  # the chat Arjun keeps continuing
NEW_CHAT_WORDS = ("new chat", "start over", "new conversation", "fresh start")
AGENT_MODEL = os.environ.get("ARJUN_VOICE_MODEL", "anthropic/claude-sonnet-5-5 --runtime openclaw")
LIVE_PROVIDER = os.environ.get("ARJUN_LIVE_PROVIDER", "google")
IDLE_CLOSE = float(os.environ.get("ARJUN_LIVE_IDLE_SECONDS", "20"))  # close the call after this much quiet
EXIT_WORDS = ("bye", "goodbye", "that's all", "that is all", "nothing else")
GATEWAY_WS = os.environ.get("OPENCLAW_GATEWAY_WS", "ws://127.0.0.1:18789")
GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN") or json.loads(
    (Path.home() / ".openclaw/openclaw.json").read_text())["gateway"]["auth"]["token"]
OVERLAY_ADDR = ("127.0.0.1", 47820)  # overlay.py
CONTROL_ADDR = ("127.0.0.1", 47821)  # from arjun_app.py: mute state, play/stop, new chat
CONSULT_TIMEOUT = 180
# With speakers: "half" mutes the mic towards Gemini while Arjun speaks (say the wake word to interrupt), as
# Bluetooth delay defeats echo cancellation; "full" lets you talk over him. Headphones are always full.
DUPLEX = os.environ.get("ARJUN_DUPLEX", "half")
ECHO_TAIL = 0.8  # Bluetooth keeps playing for a while after the buffer empties
CONSULT_PREFIX = ("[VOICE task, relayed by the live voice assistant. Answer in plain spoken English, no markdown, "
                  "brief.] ")

RATE = 16000  # mic rate
CHUNK = 1280  # 80 ms
VAD_FRAME = 512  # 32 ms, what Silero expects at 16 kHz
WAKE_CHECK_EVERY = 0.5  # transcribe the speech so far this often while someone talks
WAKE_WINDOW = 4.0  # seconds of speech Whisper looks at (and that is handed to Gemini on wake)
WAKE_COOLDOWN = 2.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("arjun-live")
logging.getLogger("faster_whisper").setLevel(logging.WARNING)  # it logs every transcription


class VoiceActivity:
    """Silero VAD on mic chunks: is the person talking right now? Keeps a call from timing out mid-sentence,
    since Gemini only reports what was said once the turn ends."""

    def __init__(self):
        self.vad, self.pending = load_silero_vad(), np.zeros(0, dtype=np.float32)

    def speaking(self, chunk: np.ndarray) -> bool:
        self.pending = np.concatenate([self.pending, chunk.astype(np.float32) / 32768])
        heard = False
        while self.pending.size >= VAD_FRAME:
            frame, self.pending = self.pending[:VAD_FRAME], self.pending[VAD_FRAME:]
            heard |= self.vad(torch.from_numpy(frame), RATE).item() > 0.5
        return heard


class WakeListener:
    """Silero VAD + Whisper: fires when someone says "hey <name>". Understands accents a small model misses."""

    GREETINGS = {"hey", "hi", "hay", "ok", "okay", "hello"}

    def __init__(self):
        self.vad = load_silero_vad()
        self.stt = WhisperModel(STT_MODEL, device="cuda", compute_type="float16")
        self.reset()

    def reset(self):
        self.vad.reset_states()
        self.pending = np.zeros(0, dtype=np.float32)
        self.frames = collections.deque(maxlen=int(WAKE_WINDOW * RATE / VAD_FRAME))
        self.speaking, self.silence, self.unchecked = False, 0.0, 0.0

    def heard_wake(self, text: str) -> bool:
        words = re.findall(r"[a-z]+", text.lower())
        return any(prev in self.GREETINGS and difflib.SequenceMatcher(None, w, WAKE_NAME).ratio() >= 0.6
                   for prev, w in zip(words, words[1:]))

    def check(self):
        audio = np.concatenate(self.frames)
        segments, _ = self.stt.transcribe(audio, language="en", beam_size=1, vad_filter=False)
        text = " ".join(s.text for s in segments).strip()
        if self.heard_wake(text):
            log.info("wake phrase heard: %s", text)
            return audio
        return None

    def feed(self, chunk: np.ndarray):
        """Return the recent speech (float32, 16 kHz) when it contains the wake phrase, else None."""
        self.pending = np.concatenate([self.pending, chunk.astype(np.float32) / 32768])
        ended = False
        while self.pending.size >= VAD_FRAME:
            frame, self.pending = self.pending[:VAD_FRAME], self.pending[VAD_FRAME:]
            self.frames.append(frame)  # keeps a little audio from before speech starts, too
            if self.vad(torch.from_numpy(frame), RATE).item() > 0.5:
                self.speaking, self.silence = True, 0.0
            elif self.speaking:
                self.silence += VAD_FRAME / RATE
            if self.speaking:
                self.unchecked += VAD_FRAME / RATE
                if self.silence > 0.6:
                    self.speaking, ended = False, True
        if (ended or self.unchecked >= WAKE_CHECK_EVERY) and self.unchecked > 0:
            self.unchecked = 0.0
            return self.check()
        return None


_overlay_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def overlay(state: str = None, level: float = None):
    msg = {k: v for k, v in (("state", state), ("level", level)) if v is not None}
    try:
        _overlay_sock.sendto(json.dumps(msg).encode(), OVERLAY_ADDR)
    except OSError:
        pass


def output_is_headphones() -> bool:
    """True when the default output is headphones/a headset (PipeWire device.form_factor)."""
    try:
        sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=2).stdout
        sinks = subprocess.run(["pactl", "list", "sinks"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    block = sinks.split(f"Name: {sink.strip()}\n", 1)[-1].split("\nSink #", 1)[0] if sink.strip() else ""
    return re.search(r'device\.form_factor = "(headphone|headset)"', block) is not None


def same_task(a: str, b: str) -> bool:
    """Gemini re-asks a cancelled task in different words; compare the words, not their order."""
    wa, wb = (set(re.findall(r"[a-z0-9']+", s.lower())) for s in (a, b))
    return len(wa & wb) / max(1, len(wa | wb)) >= 0.4


class Gateway:
    """Minimal OpenClaw Gateway client: request/response plus event listeners."""

    def __init__(self):
        self.ws, self.pending, self.listeners = None, {}, []

    async def connect(self):
        self.ws = await websockets.connect(GATEWAY_WS, max_size=None)
        challenge = json.loads(await self.ws.recv())
        assert challenge.get("event") == "connect.challenge", challenge
        await self.ws.send(json.dumps({"type": "req", "id": "connect", "method": "connect", "params": {
            "minProtocol": 4, "maxProtocol": 4,
            "client": {"id": "cli", "version": "1.0.0", "platform": "linux", "mode": "cli"},
            "role": "operator", "scopes": ["operator.read", "operator.write"],
            "auth": {"token": GATEWAY_TOKEN}}}))
        res = json.loads(await self.ws.recv())
        if not res.get("ok"):
            raise RuntimeError(f"gateway auth failed: {res.get('error')}")
        asyncio.create_task(self._reader())

    async def _reader(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            if msg.get("type") == "res":
                fut = self.pending.pop(msg.get("id"), None)
                if fut and not fut.done():
                    if msg.get("ok"):
                        fut.set_result(msg.get("payload"))
                    else:
                        fut.set_exception(RuntimeError(msg.get("error", {}).get("message", "request failed")))
            elif msg.get("type") == "event":
                for listener in list(self.listeners):
                    listener(msg["event"], msg.get("payload") or {})
        log.error("gateway connection closed")
        os._exit(1)  # systemd restarts us with a fresh connection

    async def request(self, method: str, params: dict, timeout: float = 30):
        rid = uuid.uuid4().hex
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps({"type": "req", "id": rid, "method": method, "params": params}))
        return await asyncio.wait_for(fut, timeout)

    async def send(self, method: str, params: dict):
        """Fire-and-forget request (responses are ignored)."""
        await self.ws.send(json.dumps({"type": "req", "id": "ff-" + uuid.uuid4().hex, "method": method,
                                       "params": params}))


class Player:
    """24 kHz PCM16 playback with an instantly clearable buffer (for barge-in)."""

    def __init__(self, rate: int):
        self.rate, self.buf, self.lock, self.last_played = rate, bytearray(), threading.Lock(), 0.0
        self.stream = sd.OutputStream(samplerate=rate, channels=1, dtype="int16", callback=self._callback)
        self.stream.start()

    def _callback(self, outdata, frames, t, status):
        need = frames * 2
        with self.lock:
            chunk, self.buf[:need] = bytes(self.buf[:need]), b""
        if chunk:
            self.last_played = time.time()
        outdata[:] = np.frombuffer(chunk.ljust(need, b"\0"), dtype=np.int16).reshape(-1, 1)

    def speaking(self) -> bool:
        """True while audio is queued or may still be coming out of a (Bluetooth-delayed) speaker."""
        return self.queued_seconds() > 0 or time.time() - self.last_played < ECHO_TAIL

    def write(self, pcm: bytes):
        with self.lock:
            self.buf += pcm

    def clear(self):
        with self.lock:
            self.buf.clear()

    def queued_seconds(self) -> float:
        with self.lock:
            return len(self.buf) / 2 / self.rate

    def level(self) -> float:
        with self.lock:
            head = np.frombuffer(bytes(self.buf[:2400]), dtype=np.int16)
        return float(np.sqrt(np.mean((head / 32768.0) ** 2))) * 4 if head.size else 0.0


class LiveCall:
    """One realtime Talk call over the Gateway relay."""

    def __init__(self, gw: Gateway, key: str):
        self.gw, self.key, self.id, self.player, self.closed = gw, key, None, None, False
        self.new_chat = False  # the person asked for a new chat; the next call uses a fresh session
        self.last_activity, self.tools = time.time(), {}
        self.consult_runs = {}  # runId -> future(text)
        self.active = None  # the agent task in progress: {"question", "run_id", "fut"}
        self.turn_id = self.interrupted_turn = None

    async def start(self, preroll: list):
        overlay("listening")
        res = await self.gw.request("talk.session.create", {
            "sessionKey": self.key, "provider": LIVE_PROVIDER, "mode": "realtime",
            "transport": "gateway-relay", "brain": "agent-consult"})
        self.id, audio = res["relaySessionId"], res["audio"]
        self.in_rate = audio["inputSampleRateHz"]
        self.player = Player(audio["outputSampleRateHz"])
        self.gw.listeners.append(self.on_event)
        log.info("call open (%s %s, %d Hz)", res.get("provider"), res.get("model"), self.in_rate)
        for chunk in preroll:
            await self.send_audio(chunk)

    async def send_audio(self, chunk16k: np.ndarray):
        pcm = ss.resample_poly(chunk16k.astype(np.float32), self.in_rate // 8000, RATE // 8000)
        pcm = np.clip(pcm, -32768, 32767).astype(np.int16).tobytes()
        await self.gw.send("talk.session.appendAudio", {
            "sessionId": self.id, "audioBase64": base64.b64encode(pcm).decode(),
            "timestamp": int(time.time() * 1000)})

    def on_event(self, event: str, p: dict):
        if event == "chat" and p.get("runId") in self.consult_runs:
            fut = self.consult_runs[p["runId"]]
            if p.get("state") == "final" and not fut.done():
                fut.set_result("".join(b.get("text", "") for b in p.get("message", {}).get("content", [])
                                       if b.get("type") == "text") or "OpenClaw finished with no text.")
            elif p.get("state") in ("error", "aborted") and not fut.done():
                fut.set_exception(RuntimeError(p.get("errorMessage", p["state"])))
            return
        if event != "talk.event" or p.get("relaySessionId") != self.id or self.closed:
            return
        kind = p.get("type")
        turn_id = (p.get("talkEvent") or {}).get("turnId")
        if kind == "audio" and p.get("audioBase64"):
            if turn_id and turn_id == self.interrupted_turn:
                return  # rest of a reply the person cut off
            self.turn_id = turn_id
            self.player.write(base64.b64decode(p["audioBase64"]))
            self.last_activity = time.time()
        elif kind == "clear":  # provider detected barge-in: drop what's queued
            self.player.clear()
        elif kind == "mark" and p.get("markName"):
            asyncio.create_task(self.ack_mark(p["markName"]))
        elif kind == "transcript" and p.get("final") and p.get("text"):
            log.info("%s: %s", p.get("role"), p["text"])
            self.last_activity = time.time()
            said, words = p["text"].lower(), len(p["text"].split())
            if p.get("role") == "user" and any(w in said for w in NEW_CHAT_WORDS) and words <= 6:
                self.new_chat = True
                overlay("new")
                self.exit_at = time.time() + 4  # let Arjun acknowledge, then hang up
            elif p.get("role") == "user" and any(w in said for w in EXIT_WORDS) and words <= 4:
                self.exit_at = time.time() + 3  # let the goodbye play, then hang up
        elif kind == "toolCall":
            self.tools[p["callId"]] = asyncio.create_task(self.handle_tool(p))
        elif kind == "toolCallCancelled":
            task = self.tools.pop(p.get("callId"), None)
            if task:
                task.cancel()
        elif kind == "error":
            log.error("relay error: %s", p.get("message"))
        elif kind == "close":
            log.info("relay closed (%s)", p.get("reason"))
            self.closed = True

    async def interrupt(self):
        """Wake word said while Arjun talks: stop his voice now and let the person speak."""
        self.interrupted_turn = self.turn_id
        self.player.clear()
        self.player.last_played = 0.0
        await self.gw.send("talk.session.cancelOutput", {
            "sessionId": self.id, "reason": "barge-in", **({"turnId": self.turn_id} if self.turn_id else {})})
        self.last_activity = time.time()

    async def ack_mark(self, name: str):
        await asyncio.sleep(self.player.queued_seconds())  # ack once the audio before the mark has played
        if not self.closed:
            await self.gw.send("talk.session.acknowledgeMark", {"sessionId": self.id, "markName": name})

    async def submit(self, call_id: str, result: dict, will_continue: bool = False):
        await self.gw.request("talk.session.submitToolResult", {
            "sessionId": self.id, "callId": call_id, "result": result,
            **({"options": {"willContinue": True}} if will_continue else {})})

    async def handle_tool(self, p: dict):
        call_id, name = p["callId"], p.get("name")
        args = p.get("args") or {}
        args = json.loads(args) if isinstance(args, str) else args
        log.info("tool %s %s", name, json.dumps(args)[:200])
        try:
            if name == "openclaw_agent_control":
                if args.get("mode") == "status" and self.active:  # the gateway can't see our chat.send task
                    await self.submit(call_id, {
                        "status": "done" if self.active["fut"].done() else "running",
                        "task": self.active["question"],
                        "elapsedSeconds": int(time.time() - self.active["started"]),
                        "instruction": "Tell the person briefly, then call openclaw_agent_consult again with "
                                       "the same question to wait for and receive the result."})
                    return
                if args.get("mode") == "cancel" and self.active:
                    await self.gw.send("chat.abort", {"sessionKey": self.key, "runId": self.active["run_id"]})
                    log.info("cancelled task %s", self.active["run_id"])
                    self.active = None
                res = await self.gw.request("talk.session.steer", {
                    "sessionId": self.id, "sessionKey": self.key, "text": args.get("text", ""),
                    **({"mode": args["mode"]} if args.get("mode") else {})})
                await self.submit(call_id, res)
                return
            if name != "openclaw_agent_consult":
                await self.submit(call_id, {"error": f'Tool "{name}" not available'})
                return
            overlay("thinking")
            if p.get("forced"):  # Gemini Live rejects interim results; only forced calls need one
                await self.submit(call_id, {"status": "working", "tool": name, "message":
                                            "Tell the person briefly that you are checking, then wait for the "
                                            "final OpenClaw result before answering with the actual result."}, True)
            task = await self.consult(args.get("question") or json.dumps(args))
            try:
                # shield: Gemini cancels its tool call when the person talks; the agent task keeps running
                text = await asyncio.wait_for(asyncio.shield(task["fut"]), CONSULT_TIMEOUT)
            except asyncio.CancelledError:
                log.info("tool call cancelled; task %s keeps running", task["run_id"])
                raise
            if self.active is task:
                self.active = None
            log.info("consult done: %s", text[:120])
            await self.submit(call_id, {"result": text})
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("tool %s failed: %s", name, e)
            await self.submit(call_id, {"error": str(e)})
        finally:
            self.tools.pop(call_id, None)
            self.last_activity = time.time()

    async def consult(self, question: str) -> dict:
        """Start an agent task, or reattach to the running one when Gemini re-asks the same thing."""
        if self.active and same_task(question, self.active["question"]):
            log.info("reattached to task %s", self.active["run_id"])
            return self.active
        res = await self.gw.request("chat.send", {
            "sessionKey": self.key, "message": CONSULT_PREFIX + question, "deliver": False,
            "timeoutMs": CONSULT_TIMEOUT * 1000, "idempotencyKey": str(uuid.uuid4())})
        fut = asyncio.get_running_loop().create_future()
        self.consult_runs[res["runId"]] = fut
        fut.add_done_callback(lambda f: f.exception())  # an unclaimed failure must not warn
        self.active = {"question": question, "run_id": res["runId"], "fut": fut, "started": time.time()}
        log.info("started task %s: %s", res["runId"], question[:100])
        return self.active

    def finished(self) -> bool:
        if self.closed or time.time() > getattr(self, "exit_at", float("inf")):
            return True
        busy = self.tools or self.player.queued_seconds() > 0
        return not busy and time.time() - self.last_activity > IDLE_CLOSE

    def update_overlay(self):
        if self.player.queued_seconds() > 0:
            overlay("speaking", self.player.level())
        elif self.tools:
            overlay("thinking")
        else:
            overlay("listening")

    async def close(self):
        self.closed = True
        for task in list(self.tools.values()):
            task.cancel()
        if self.active and not self.active["fut"].done():
            # Let the aborted run finish before closing: closing writes the call's transcript into the same
            # chat, and a run still rewriting it fails with "SQLite transcript changed while preparing rewrite".
            await self.gw.send("chat.abort", {"sessionKey": self.key, "runId": self.active["run_id"]})
            await asyncio.wait([self.active["fut"]], timeout=5)
        try:
            await self.gw.request("talk.session.close", {"sessionId": self.id}, timeout=8)
        except Exception as e:
            log.warning("close failed: %s", e)
        if self.player:
            self.player.stream.close()
        if self.on_event in self.gw.listeners:
            self.gw.listeners.remove(self.on_event)
        overlay("idle")
        log.info("call closed")


async def pin_model(gw: Gateway, key: str):
    try:
        await gw.request("chat.send", {"sessionKey": key, "message": f"/model {AGENT_MODEL} -s",
                                       "deliver": False, "idempotencyKey": str(uuid.uuid4())})
    except Exception as e:
        log.warning("could not pin %s to %s: %s", key, AGENT_MODEL, e)


async def main():
    stopping = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopping.set)
    gw = Gateway()
    await gw.connect()
    try:
        session = json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        session = {"key": "voice-live"}
    log.info("continuing chat %s", session["key"])
    await pin_model(gw, session["key"])

    async def start_new_chat():
        overlay("new")  # gold flash
        session["key"] = "voice-live-" + time.strftime("%Y%m%d-%H%M%S")
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(session))
        await pin_model(gw, session["key"])
        log.info("new chat: %s", session["key"])
        await asyncio.sleep(1)
        overlay("idle")

    loop = asyncio.get_running_loop()
    mic: asyncio.Queue = asyncio.Queue()
    listener, voice = WakeListener(), VoiceActivity()

    def open_mic(source: str):
        def on_audio(indata, frames, t, status):
            gain = MIC_GAIN if source == MIC else 1.0
            chunk = np.clip(indata[:, 0] * gain, -32768, 32767).astype(np.int16)
            loop.call_soon_threadsafe(mic.put_nowait, (source, chunk))
        os.environ["PULSE_SOURCE"] = source  # read when PortAudio opens the pulse device
        stream = sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK, device="pulse",
                                callback=on_audio)
        stream.start()
        return stream

    class Controls(asyncio.DatagramProtocol):
        """Mute state and play/stop/new-chat commands from the overlay and tray menu."""
        muted = start = stop = new_chat = False

        def datagram_received(self, data, addr):
            msg = json.loads(data)
            if msg.get("stop"):
                Controls.stop = True
            if msg.get("new_chat"):
                Controls.new_chat = True
            if msg.get("start") and not Controls.muted:  # play does what "Hey Arjun" does
                Controls.start = True
            muted = bool(msg.get("muted", Controls.muted))
            if muted != Controls.muted:
                log.info("muted" if muted else "unmuted")
                Controls.muted = muted
                listener.reset()

    await loop.create_datagram_endpoint(Controls, local_addr=CONTROL_ADDR)
    streams = [open_mic(source) for source in {MIC, RAW_MIC}]  # referenced so they stay open  # noqa: F841
    call, last_wake, headphones, checked, was_speaking = None, 0.0, None, 0.0, False
    log.info("listening for 'hey %s'", WAKE_NAME)
    while True:
        source, chunk = await mic.get()
        if stopping.is_set():  # SIGTERM from arjun_app.py / systemd: hang up cleanly, then exit
            if call:
                await call.close()
            return
        if time.time() - checked > 3:
            checked, was = time.time(), headphones
            headphones = await asyncio.to_thread(output_is_headphones)
            if headphones != was:
                log.info("output is %s: using %s", "headphones" if headphones else "speakers",
                         RAW_MIC if headphones else MIC)
        if source != (RAW_MIC if headphones else MIC):
            continue
        if Controls.new_chat:  # from the tray menu
            Controls.new_chat = False
            if call:
                call.new_chat, call.exit_at = True, time.time()  # hang up; the chat switches below
            else:
                await start_new_chat()
        if Controls.stop:
            Controls.stop = False
            if call:  # stop button: silence Arjun, cancel his task and end the call
                log.info("stopped from the overlay")
                await call.close()
                call = None
                listener.reset()
                continue
        if Controls.muted:
            if call is None:
                continue  # no wake detection while muted
            chunk = np.zeros_like(chunk)  # an open call hears silence; it ends on its idle timeout
        if call is None:
            speech = await asyncio.to_thread(listener.feed, chunk)
            if speech is None and not Controls.start:
                continue
            if speech is None:
                log.info("started from the overlay")
            Controls.start = False
            last_wake = time.time()
            listener.reset()
            call = LiveCall(gw, session["key"])
            # the wake phrase and whatever followed it (nothing when started from the overlay)
            pcm = np.zeros(0, np.int16) if speech is None else (speech * 32768).astype(np.int16)
            try:
                await call.start([pcm[i:i + CHUNK] for i in range(0, pcm.size, CHUNK)])
            except Exception as e:
                log.error("could not open call: %s", e)
                call = None
                overlay("idle")
            continue
        speaking = DUPLEX == "half" and not headphones and not Controls.muted and call.player.speaking()
        if speaking:
            if not was_speaking:
                listener.reset()
            if await asyncio.to_thread(listener.feed, chunk) is not None and time.time() - last_wake > WAKE_COOLDOWN:
                log.info("interrupted by wake phrase")
                last_wake = time.time()
                await call.interrupt()
                listener.reset()
            else:
                chunk = np.zeros_like(chunk)  # keep Gemini from hearing its own voice
        was_speaking = speaking
        if not Controls.muted and voice.speaking(chunk):
            call.last_activity = time.time()  # still talking, even if Gemini hasn't reported it yet
        await call.send_audio(chunk)
        call.update_overlay()
        if call.finished():
            await call.close()
            if call.new_chat:
                await start_new_chat()
            call = None
            overlay("idle")
            listener.reset()


if __name__ == "__main__":
    asyncio.run(main())
