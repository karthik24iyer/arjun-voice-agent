"""Bottom-of-screen wave overlay for Hey Arjun (Wayland layer-shell, 50% transparent), with play/stop and mute buttons.

Part of arjun_app.py. Always visible: a grey wave while Arjun isn't listening, coloured while in a call, flat while
muted or stopped. Only the buttons take clicks; the wave is click-through.
Listens on UDP for {"state": "idle|listening|thinking|speaking|new", "level": 0..1} from the assistant, and
sends back {"muted": bool} (on click and every 2 s, so a restarted assistant picks it up) and {"start": true}
(play, like saying "Hey Arjun") or {"stop": true} (stop, while a call is running).
Runs with the system python3 (needs gi, GTK3, gir1.2-gtklayershell-0.1 and python3-gi-cairo).
"""
import json
import math
import socket
import time

import cairo
import gi

gi.require_version("Gdk", "3.0")
gi.require_version("Gtk", "3.0")
gi.require_version("GtkLayerShell", "0.1")
from gi.repository import Gdk, GLib, Gtk, GtkLayerShell  # noqa: E402

PORT = 47820  # state from the assistant
CONTROL_ADDR = ("127.0.0.1", 47821)  # mute state and play/stop/new-chat commands to the assistant
WAVE_WIDTH, HEIGHT, BOTTOM_MARGIN = 280, 90, 48
BUTTON_SIZE, GAP = 36, 12
CLICK_DEBOUNCE = 0.25  # seconds
OPACITY = 0.5
COLORS = {  # (r, g, b) per state
    "idle": (0.6, 0.6, 0.6),  # not listening: waiting for "Hey Arjun"
    "listening": (0.25, 0.65, 1.0),
    "thinking": (0.65, 0.4, 1.0),
    "speaking": (0.25, 0.9, 0.55),
    "new": (1.0, 0.8, 0.25),  # gold flash when a new chat starts
}


class Overlay(Gtk.Window):
    def __init__(self):
        super().__init__()
        self.state, self.target, self.level, self.phase, self.muted = "idle", 0.0, 0.0, 0.0, False
        self.offline = False  # the assistant isn't running
        self.last_click = 0.0
        self.color = COLORS["idle"]
        top = (HEIGHT - BUTTON_SIZE) // 2
        self.stop_button = (WAVE_WIDTH + GAP, top, BUTTON_SIZE, BUTTON_SIZE)  # x, y, w, h
        self.button = (WAVE_WIDTH + 2 * GAP + BUTTON_SIZE, top, BUTTON_SIZE, BUTTON_SIZE)

        GtkLayerShell.init_for_window(self)
        GtkLayerShell.set_layer(self, GtkLayerShell.Layer.OVERLAY)
        GtkLayerShell.set_anchor(self, GtkLayerShell.Edge.BOTTOM, True)
        GtkLayerShell.set_margin(self, GtkLayerShell.Edge.BOTTOM, BOTTOM_MARGIN)
        GtkLayerShell.set_keyboard_mode(self, GtkLayerShell.KeyboardMode.NONE)
        GtkLayerShell.set_namespace(self, "arjun-overlay")

        self.set_size_request(WAVE_WIDTH + 2 * (GAP + BUTTON_SIZE), HEIGHT)
        self.set_app_paintable(True)
        self.set_visual(self.get_screen().get_rgba_visual())
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.connect("draw", self.on_draw)
        self.connect("button-press-event", self.on_click)
        # only the buttons take clicks; everything else passes through to the windows below
        self.connect("realize", lambda w: w.input_shape_combine_region(
            cairo.Region([cairo.RectangleInt(*self.stop_button), cairo.RectangleInt(*self.button)])))

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", PORT))
        self.sock.setblocking(False)
        GLib.io_add_watch(self.sock.fileno(), GLib.IO_IN, self.on_message)
        GLib.timeout_add(16, self.tick)
        GLib.timeout_add_seconds(2, self.send_mute)
        self.show_all()

    def send(self, msg: dict):
        self.sock.sendto(json.dumps(msg).encode(), CONTROL_ADDR)

    def send_mute(self):
        self.send({"muted": self.muted})
        return True

    def set_muted(self, muted: bool):
        self.muted = muted
        self.send_mute()
        self.queue_draw()

    def talk_or_stop(self):
        """Play while idle (like saying "Hey Arjun"), stop while a call runs."""
        self.send({"start" if self.state == "idle" else "stop": True})

    def on_click(self, _, event):
        # COSMIC delivers each click twice (plus double/triple-click events), which would toggle mute back off
        if event.type != Gdk.EventType.BUTTON_PRESS or time.monotonic() - self.last_click < CLICK_DEBOUNCE:
            return
        self.last_click = time.monotonic()

        def inside(rect):
            x, y, w, h = rect
            return event.button == 1 and x <= event.x <= x + w and y <= event.y <= y + h
        if inside(self.button):
            self.set_muted(not self.muted)
        elif inside(self.stop_button):
            self.talk_or_stop()

    def on_message(self, *_):
        try:
            while True:
                msg = json.loads(self.sock.recv(1024))
                if "state" in msg and msg["state"] != self.state:
                    self.state = msg["state"]
                    self.color = COLORS.get(self.state, self.color)
                if "level" in msg:
                    self.target = min(1.0, float(msg["level"]))
        except BlockingIOError:
            pass
        return True

    def tick(self):
        if self.state == "idle":
            self.target = 0.04
        elif self.state == "thinking":  # no audio while thinking: slow breathing
            self.target = 0.25 + 0.15 * math.sin(self.phase * 0.6)
        self.level += (self.target - self.level) * 0.25
        self.phase += 0.04 if self.state == "idle" else 0.12
        self.queue_draw()
        return True

    def on_draw(self, _, cr):
        cr.set_operator(cairo.OPERATOR_SOURCE)
        cr.set_source_rgba(0, 0, 0, 0)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)
        self.draw_wave(cr)
        self.draw_stop_button(cr)
        self.draw_button(cr)

    def draw_wave(self, cr):
        mid = HEIGHT / 2
        flat = self.muted or self.offline
        r, g, b = COLORS["idle"] if flat else self.color
        amp = 0 if flat else 4 + self.level * (HEIGHT / 2 - 8)
        for i, (speed, freq, alpha) in enumerate(((1.0, 2.0, 1.0), (1.4, 3.0, 0.6), (0.7, 1.5, 0.4))):
            cr.set_source_rgba(r, g, b, OPACITY * alpha)
            cr.set_line_width(3 - i * 0.6)
            for x in range(0, WAVE_WIDTH + 1, 4):
                taper = math.sin(math.pi * x / WAVE_WIDTH)  # fade out towards both ends
                y = mid + amp * taper * math.sin(freq * 2 * math.pi * x / WAVE_WIDTH + self.phase * speed + i)
                (cr.move_to if x == 0 else cr.line_to)(x, y)
            cr.stroke()

    def draw_stop_button(self, cr):
        x, y, w, h = self.stop_button
        cx, cy = x + w / 2, y + h / 2
        cr.set_source_rgba(0.15, 0.15, 0.15, 0.3)
        cr.arc(cx, cy, w / 2, 0, 2 * math.pi)
        cr.fill()
        cr.set_source_rgba(1, 1, 1, OPACITY)
        if self.state == "idle":  # play triangle
            cr.move_to(cx - 4, cy - 7)
            cr.line_to(cx + 7, cy)
            cr.line_to(cx - 4, cy + 7)
            cr.close_path()
        else:  # stop square
            cr.rectangle(cx - 6, cy - 6, 12, 12)
        cr.fill()

    def draw_button(self, cr):
        x, y, w, h = self.button
        cx, cy = x + w / 2, y + h / 2
        cr.set_source_rgba(0.85, 0.2, 0.2, OPACITY) if self.muted else cr.set_source_rgba(0.15, 0.15, 0.15, 0.3)
        cr.arc(cx, cy, w / 2, 0, 2 * math.pi)
        cr.fill()
        # microphone: capsule, cradle and stand
        cr.set_source_rgba(1, 1, 1, OPACITY)
        cr.set_line_width(2)
        cr.arc(cx, cy - 6, 3.5, math.pi, 0)
        cr.line_to(cx + 3.5, cy + 1)
        cr.arc(cx, cy + 1, 3.5, 0, math.pi)
        cr.close_path()
        cr.fill()
        cr.arc(cx, cy, 7, 0, math.pi)
        cr.stroke()
        cr.move_to(cx, cy + 7)
        cr.line_to(cx, cy + 11)
        cr.stroke()
        if self.muted:
            cr.move_to(cx - 9, cy - 9)
            cr.line_to(cx + 9, cy + 9)
            cr.stroke()

