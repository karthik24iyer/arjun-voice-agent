#!/usr/bin/env python3
"""Arjun desktop app: tray icon + wave overlay, running the voice engine (arjun_live.py) as a supervised child.

Runs with the system python3 (gi, GTK3, gir1.2-gtklayershell-0.1, gir1.2-ayatanaappindicator3-0.1,
python3-gi-cairo); the engine runs in the project's .venv. Installed as the `arjun` systemd user service.
"""
import signal
import subprocess
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("AyatanaAppIndicator3", "0.1")
from gi.repository import AyatanaAppIndicator3 as AppIndicator, GLib, Gtk  # noqa: E402

from overlay import Overlay  # noqa: E402

HERE = Path(__file__).resolve().parent
ENGINE = [str(HERE / ".venv/bin/python"), "-u", str(HERE / "arjun_live.py")]
SERVICE = "arjun"
RESPAWN_SECONDS = 3
STATUS = {"idle": "Listening for “Hey Arjun”", "listening": "In a call: listening", "thinking": "In a call: working",
          "speaking": "In a call: speaking", "new": "Starting a new chat"}


class Engine:
    """Keeps arjun_live.py running while wanted; its output goes to this service's journal."""

    def __init__(self):
        self.proc, self.wanted = None, False
        GLib.timeout_add_seconds(RESPAWN_SECONDS, self.watchdog)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        self.wanted = True
        if not self.running():
            self.proc = subprocess.Popen(ENGINE, cwd=HERE)

    def stop(self):
        self.wanted = False
        if self.running():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)  # it hangs up cleanly first (aborting a running task can take a while)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def restart(self):
        self.stop()
        self.start()

    def watchdog(self):
        if self.wanted and not self.running():
            print(f"engine exited ({self.proc.returncode if self.proc else '?'}); restarting", flush=True)
            self.start()
        return True


class Tray:
    def __init__(self, overlay: Overlay, engine: Engine):
        self.overlay, self.engine, self.syncing = overlay, engine, False
        self.indicator = AppIndicator.Indicator.new("arjun", "audio-input-microphone-symbolic",
                                                    AppIndicator.IndicatorCategory.APPLICATION_STATUS)
        self.indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
        self.indicator.set_title("Arjun")

        menu = Gtk.Menu()
        self.status = self.item(menu, "", None)
        self.status.set_sensitive(False)
        menu.append(Gtk.SeparatorMenuItem())
        self.talk = self.item(menu, "Talk now", lambda _: overlay.talk_or_stop())
        self.mute = self.check(menu, "Mute microphone", lambda w: overlay.set_muted(w.get_active()))
        self.item(menu, "Start a new chat", lambda _: overlay.send({"new_chat": True}))
        menu.append(Gtk.SeparatorMenuItem())
        self.wave = self.check(menu, "Show wave", lambda w: overlay.show_all() if w.get_active() else overlay.hide())
        self.autostart = self.check(menu, "Start at login", self.set_autostart)
        menu.append(Gtk.SeparatorMenuItem())
        self.item(menu, "Restart Arjun", lambda _: engine.restart())
        self.power = self.item(menu, "Stop Arjun", self.toggle_engine)
        self.item(menu, "View logs", lambda _: subprocess.Popen(
            ["x-terminal-emulator", "-e", "journalctl", "--user", "-u", SERVICE, "-f", "-n", "200"]))
        menu.append(Gtk.SeparatorMenuItem())
        self.item(menu, "Quit", self.quit)
        menu.show_all()
        self.indicator.set_menu(menu)

        self.sync(initial=True)
        GLib.timeout_add(500, self.sync)

    @staticmethod
    def item(menu, label, handler):
        item = Gtk.MenuItem(label=label)
        if handler:
            item.connect("activate", handler)
        menu.append(item)
        return item

    def check(self, menu, label, handler):
        item = Gtk.CheckMenuItem(label=label)
        item.connect("toggled", lambda w: None if self.syncing else handler(w))
        menu.append(item)
        return item

    def sync(self, initial=False):
        """Reflect the current state in the menu, icon and overlay."""
        running, state = self.engine.running(), self.overlay.state
        self.overlay.offline = not running
        self.syncing = True
        self.mute.set_active(self.overlay.muted)
        if initial:
            self.wave.set_active(True)
            enabled = subprocess.run(["systemctl", "--user", "is-enabled", "--quiet", SERVICE]).returncode == 0
            self.autostart.set_active(enabled)
        self.syncing = False
        if not running:
            self.status.set_label("Arjun is stopped")
        else:
            self.status.set_label("Muted" if self.overlay.muted and state == "idle" else STATUS.get(state, state))
        self.talk.set_label("Talk now" if state == "idle" else "End call")
        self.talk.set_sensitive(running and not (self.overlay.muted and state == "idle"))
        self.power.set_label("Stop Arjun" if self.engine.wanted else "Start Arjun")
        icon = ("media-playback-stop-symbolic" if not running else
                "microphone-sensitivity-muted-symbolic" if self.overlay.muted else "audio-input-microphone-symbolic")
        self.indicator.set_icon_full(icon, "Arjun")
        return True

    def toggle_engine(self, _):
        if self.engine.wanted:
            self.engine.stop()
            self.overlay.state = "idle"  # the engine can't reset the overlay once it's gone
        else:
            self.engine.start()

    def set_autostart(self, item):
        subprocess.run(["systemctl", "--user", "enable" if item.get_active() else "disable", SERVICE])

    def quit(self, _=None):
        self.engine.stop()
        Gtk.main_quit()


def main():
    overlay, engine = Overlay(), Engine()
    engine.start()
    tray = Tray(overlay, engine)
    for sig in (signal.SIGTERM, signal.SIGINT):  # systemd stop: take the engine down with us
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, sig, lambda *_: tray.quit() or GLib.SOURCE_REMOVE)
    Gtk.main()


if __name__ == "__main__":
    main()
