"""Control the user's Android phone (e.g. Galaxy S25) over Wi-Fi with adb.

Setup (once, by the user):
  1. Phone: Settings → About phone → Software information → tap "Build number" 7 times.
  2. Phone: Settings → Developer options → Wireless debugging → on (same Wi-Fi as the Mac).
     If it won't turn on: Settings → Security and privacy → Auto Blocker → off.
  3. Mac:   adb pair <ip:port>   (from "Pair device with pairing code"), then adb connect <ip:port>
  4. Mac:   security add-generic-password -a jarvis -s jarvis-phone-pin -w
            (type your phone PIN when asked; it goes into the macOS Keychain, encrypted)

Safety: unlock refuses while the Mac is locked, asks for a spoken yes, only types the PIN when the
phone is verifiably showing its lock screen, never retries, and cools down after a failure (Samsung
locks you out after repeated wrong PINs). The PIN is read from the Keychain at the moment of use and
is never logged or sent to the AI model.
"""

import difflib
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from jarvis import log
from jarvis.tools.registry import Tool, params

KEYCHAIN_SERVICE = "jarvis-phone-pin"
KEYCHAIN_ACCOUNT = "jarvis"
FAILURE_COOLDOWN_SECONDS = 300
_EXTRA_PATHS = [
    Path.home() / "Library" / "Android" / "sdk" / "platform-tools",  # Android Studio
    Path("/opt/homebrew/bin"),  # Homebrew (Apple Silicon)
    Path("/usr/local/bin"),
]
SETUP_HINT = (
    "Phone isn't connected. On the phone turn on Developer options → Wireless debugging (same Wi-Fi as the "
    "Mac), then pair with `adb pair`. Full steps: README, 'Phone'."
)


def find_tool(name: str, override: str = "") -> str | None:
    """Locate a command even under launchd's minimal PATH."""
    if override:
        return override if Path(override).exists() else None
    found = shutil.which(name)
    if found:
        return found
    for folder in _EXTRA_PATHS:
        if (folder / name).exists():
            return str(folder / name)
    return None


def mac_is_locked() -> bool:
    """True if the Mac's screen is locked (or we can't tell, which we treat as locked)."""
    try:
        out = subprocess.run(["ioreg", "-n", "Root", "-d1"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    match = re.search(r'"IOConsoleLocked"\s*=\s*(Yes|No)', out)
    return match is None or match.group(1) == "Yes"


def read_pin_from_keychain() -> str | None:
    result = subprocess.run(
        ["security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True, text=True, timeout=10,
    )
    pin = result.stdout.strip()
    return pin if result.returncode == 0 and pin else None


def parse_keyguard_showing(dumpsys: str) -> bool | None:
    """Is the lock screen up? Looks for the flags various Android versions print. None = can't tell."""
    for pattern in (r"mKeyguardShowing=(true|false)", r"isKeyguardShowing=(true|false)",
                    r"mShowingLockscreen=(true|false)", r"mDreamingLockscreen=(true|false)"):
        match = re.search(pattern, dumpsys)
        if match:
            return match.group(1) == "true"
    return None


def parse_mdns_connect_addresses(output: str) -> list[str]:
    """ip:port of paired phones advertising Wireless debugging (`adb mdns services`)."""
    return re.findall(r"_adb-tls-connect\._tcp\.?\s+(\d+\.\d+\.\d+\.\d+:\d+)", output)


def parse_adb_devices(output: str) -> list[str]:
    """Serials from `adb devices -l` that are ready (state 'device'), in order."""
    serials = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serials.append(parts[0])
    return serials


# Spoken name → Android package(s), for apps whose package name doesn't look like their name.
# A tuple lists alternatives (Google vs Samsung versions); the first one installed wins.
APP_ALIASES = {
    "gmail": "com.google.android.gm", "google meet": "com.google.android.apps.tachyon",
    "meet": "com.google.android.apps.tachyon", "youtube": "com.google.android.youtube",
    "youtube music": "com.google.android.apps.youtube.music", "yt music": "com.google.android.apps.youtube.music",
    "play store": "com.android.vending", "google play": "com.android.vending", "chrome": "com.android.chrome",
    "maps": "com.google.android.apps.maps", "google maps": "com.google.android.apps.maps",
    "photos": "com.google.android.apps.photos", "google photos": "com.google.android.apps.photos",
    "drive": "com.google.android.apps.docs", "google drive": "com.google.android.apps.docs",
    "docs": "com.google.android.apps.docs.editors.docs", "sheets": "com.google.android.apps.docs.editors.sheets",
    "slides": "com.google.android.apps.docs.editors.slides", "gemini": "com.google.android.apps.bard",
    "google": "com.google.android.googlequicksearchbox", "authenticator": "com.google.android.apps.authenticator2",
    "google pay": "com.google.android.apps.nbu.paisa.user", "gpay": "com.google.android.apps.nbu.paisa.user",
    "settings": "com.android.settings", "camera": "com.sec.android.app.camera",
    "gallery": "com.sec.android.gallery3d", "messages": ("com.google.android.apps.messaging", "com.samsung.android.messaging"),
    "phone": ("com.samsung.android.dialer", "com.google.android.dialer"),
    "dialer": ("com.samsung.android.dialer", "com.google.android.dialer"),
    "contacts": ("com.samsung.android.app.contacts", "com.google.android.contacts"), "clock": "com.sec.android.app.clockpackage",
    "calculator": "com.sec.android.app.popupcalculator", "calendar": "com.samsung.android.calendar",
    "my files": "com.sec.android.app.myfiles", "files": "com.sec.android.app.myfiles",
    "notes": "com.samsung.android.app.notes", "samsung notes": "com.samsung.android.app.notes",
    "whatsapp": "com.whatsapp", "instagram": "com.instagram.android", "facebook": "com.facebook.katana",
    "messenger": "com.facebook.orca", "spotify": "com.spotify.music", "netflix": "com.netflix.mediaclient",
    "x": "com.twitter.android", "twitter": "com.twitter.android", "snapchat": "com.snapchat.android",
    "telegram": "org.telegram.messenger", "linkedin": "com.linkedin.android", "discord": "com.discord",
    "outlook": "com.microsoft.office.outlook", "teams": "com.microsoft.teams", "onedrive": "com.microsoft.skydrive",
    "chatgpt": "com.openai.chatgpt", "claude": "com.anthropic.claude",
    "blinkit": "com.grofers.customerapp", "zomato": "com.application.zomato", "swiggy": "in.swiggy.android",
    "bookmyshow": "com.bt.bms", "book my show": "com.bt.bms", "cred": "com.dreamplug.androidapp",
    "phonepe": "com.phonepe.app", "paytm": "net.one97.paytm", "amazon": "in.amazon.mShop.android.shopping",
    "flipkart": "com.flipkart.android", "myntra": "com.myntra.android", "uber": "com.ubercab",
    "ola": "com.olacabs.customer", "hotstar": "in.startv.hotstar", "jiohotstar": "in.startv.hotstar",
    "my jio": "com.jio.myjio", "makemytrip": "com.makemytrip", "goibibo": "com.goibibo",
    "canva": "com.canva.editor", "chess": "com.chess",
}
_PACKAGE_RE = re.compile(r"^[a-zA-Z][\w]*(\.[\w]+)+$")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def match_app(name: str, installed: list[str]) -> tuple[str | None, list[str]]:
    """Find the installed package for a spoken app name. Returns (package or None, close candidates)."""
    name = name.strip()
    if name in installed:  # the model passed an exact package name
        return name, []
    alias = APP_ALIASES.get(name.lower().strip(), ())
    for option in (alias,) if isinstance(alias, str) else alias:
        if option in installed:
            return option, []
    target = _norm(name)
    if not target:
        return None, []
    scored = []
    for package in installed:
        segments = [s for s in package.lower().split(".") if s not in {"com", "android", "app", "apps", "google", "in", "org", "net"}]
        best = 0.0
        for seg in segments:
            seg_n = _norm(seg)
            if seg_n == target:
                best = max(best, 1.0)
            elif target in seg_n or (len(seg_n) >= 4 and seg_n in target):
                best = max(best, 0.85)
            else:
                best = max(best, difflib.SequenceMatcher(None, target, seg_n).ratio())
        scored.append((best, package))
    scored.sort(reverse=True)
    top_score = scored[0][0] if scored else 0
    exact = [p for s, p in scored if s >= 0.85]
    if len(exact) == 1 or (exact and top_score == 1.0 and scored[1][0] < 1.0):
        return exact[0], []
    return None, [p for s, p in scored[:10] if s >= 0.5]


_MDNS_SERIAL = re.compile(r"^adb-([A-Za-z0-9]+)-[^.]*\._adb-tls-connect\._tcp")


def parse_awake(dumpsys_power: str) -> bool | None:
    match = re.search(r"mWakefulness=(\w+)", dumpsys_power)
    return None if match is None else match.group(1) == "Awake"


class AndroidPhone:
    def __init__(self, serial: str = "", adb_path: str = "", clock=time.monotonic, sleep=time.sleep):
        self.adb = find_tool("adb", adb_path)
        self.serial = serial  # from settings; may be empty
        self._target = ""  # the connection we picked when serial isn't set
        self._clock, self._sleep = clock, sleep
        self._blocked_until = 0.0
        self._apps: list[str] = []
        self._apps_at = -1e9

    # --- adb plumbing ----------------------------------------------------------------

    def _run(self, *args: str, timeout: float = 10) -> subprocess.CompletedProcess:
        target = self.serial or self._target
        cmd = [self.adb] + (["-s", target] if target else []) + list(args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def shell(self, command: str, timeout: float = 10) -> str:
        result = self._run("shell", command, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"adb shell failed: {command.split()[0]}")
        return result.stdout

    def _hardware_serial(self, connection: str) -> str:
        if match := _MDNS_SERIAL.match(connection):
            return match.group(1)
        cmd = [self.adb, "-s", connection, "shell", "getprop", "ro.serialno"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        return out or connection

    def connected_devices(self) -> list[str]:
        """One connection per physical phone.

        A phone paired over Wi-Fi often shows up twice: once via `adb connect ip:port` and once via
        adb's automatic discovery ("adb-SERIAL-xyz._adb-tls-connect._tcp"). We keep one per hardware
        serial, preferring the discovered name because ip:port changes whenever Wireless debugging restarts.
        """
        connections = parse_adb_devices(self._run_global("devices", "-l").stdout)
        by_phone: dict[str, str] = {}
        for conn in connections:
            hw = self._hardware_serial(conn)
            if hw not in by_phone or _MDNS_SERIAL.match(conn):
                by_phone[hw] = conn
        return list(by_phone.values())

    def _run_global(self, *args: str) -> subprocess.CompletedProcess:
        """adb command that isn't aimed at a particular device."""
        return subprocess.run([self.adb, *args], capture_output=True, text=True, timeout=10)

    def reconnect(self) -> None:
        """Revive dropped Wi-Fi connections (phones go 'offline' while asleep) and re-find the phone."""
        self._run_global("reconnect", "offline")
        found = parse_mdns_connect_addresses(self._run_global("mdns", "services").stdout)
        for address in found:
            self._run_global("connect", address)

    def check_connected(self, attempts: int = 3) -> str | None:
        """None if exactly one usable phone is connected, else a message explaining why not."""
        if self.adb is None:
            return "Error: adb isn't installed. Install Android platform tools (brew install android-platform-tools)."
        try:
            devices = self.connected_devices()
            for _ in range(attempts - 1):
                if devices:
                    break
                self.reconnect()
                self._sleep(1.0)
                devices = self.connected_devices()
        except (OSError, subprocess.SubprocessError) as e:
            log.event("phone_error", stage="connect", error=repr(e))
            return f"Error: couldn't run adb ({e})."
        if not devices:
            raw = self._run_global("devices", "-l")
            log.event("phone_not_found", stdout=raw.stdout, stderr=raw.stderr, returncode=raw.returncode)
        if self.serial:
            return None if self.serial in devices else f"Error: {SETUP_HINT}"
        if not devices:
            return f"Error: {SETUP_HINT}"
        if len(devices) > 1:
            return "Error: several Android phones are connected; set JARVIS_PHONE_SERIAL in .env to pick one."
        self._target = devices[0]
        return None

    def keyguard_showing(self) -> bool | None:
        for command in ("dumpsys window", "dumpsys activity activities"):
            state = parse_keyguard_showing(self.shell(command, timeout=15))
            if state is not None:
                return state
        return None

    # --- actions ---------------------------------------------------------------------

    def unlock(self) -> str:
        if mac_is_locked():
            return "Error: refusing to unlock the phone while the Mac is locked."
        if self._clock() < self._blocked_until:
            wait = int(self._blocked_until - self._clock())
            return f"Error: the last unlock attempt failed, so I'm waiting {wait} more seconds to avoid a lockout."
        if problem := self.check_connected():
            return problem
        pin = read_pin_from_keychain()
        if pin is None:
            return (
                "Error: no phone PIN saved. In a terminal run: "
                f"security add-generic-password -a {KEYCHAIN_ACCOUNT} -s {KEYCHAIN_SERVICE} -w"
            )
        if not re.fullmatch(r"[A-Za-z0-9]{4,32}", pin):
            return "Error: only PINs or passwords made of letters and digits are supported."

        self.shell("input keyevent KEYCODE_WAKEUP")
        self._sleep(0.3)
        locked = self.keyguard_showing()
        if locked is False:
            return "The phone is already unlocked; I've woken the screen."
        if locked is None:
            return "Error: couldn't tell whether the phone is at its lock screen, so I didn't type the PIN."

        # Bring up the PIN pad, then confirm it's still the lock screen before typing anything.
        self.shell("wm dismiss-keyguard")
        self._sleep(0.6)
        if self.keyguard_showing() is not True:
            return "The phone unlocked without needing the PIN."
        self.shell(f"input text {shlex.quote(pin)}")
        self.shell("input keyevent KEYCODE_ENTER")
        self._sleep(1.0)

        if self.keyguard_showing() is False:
            return "Phone unlocked."
        self._blocked_until = self._clock() + FAILURE_COOLDOWN_SECONDS
        return (
            "Error: the phone is still locked after entering the PIN. I won't try again for 5 minutes "
            "(repeated wrong PINs lock the phone). Check the saved PIN, or that the phone wasn't "
            "restarted (then it needs the PIN typed by hand once)."
        )

    def installed_apps(self) -> list[str]:
        """Launchable apps on the phone (cached for 5 minutes)."""
        if not self._apps or self._clock() - self._apps_at > 300:
            out = self.shell(
                "cmd package query-activities --brief -a android.intent.action.MAIN "
                "-c android.intent.category.LAUNCHER",
                timeout=20,
            )
            self._apps = sorted({line.strip().split("/")[0] for line in out.splitlines() if "/" in line})
            self._apps_at = self._clock()
        return self._apps

    def open_app(self, app: str) -> str:
        if problem := self.check_connected():
            return problem
        package, candidates = match_app(app, self.installed_apps())
        if package is None:
            if candidates:
                return (
                    f"Error: not sure which app '{app}' is. Installed candidates: {', '.join(candidates)}. "
                    "Call open_phone_app again with the right package name, or ask the user."
                )
            return f"Error: no app matching '{app}' is installed on the phone."
        self.shell("input keyevent KEYCODE_WAKEUP")
        if self.keyguard_showing():
            return "Error: the phone is locked. Unlock it with unlock_phone first, then open the app."
        resolved = self.shell(f"cmd package resolve-activity --brief -c android.intent.category.LAUNCHER {package}")
        component = resolved.strip().splitlines()[-1] if "/" in resolved else ""
        if component:
            self.shell(f"am start -n {shlex.quote(component)}")
        else:
            self.shell(f"monkey -p {shlex.quote(package)} -c android.intent.category.LAUNCHER 1")
        return f"Opened {package} on the phone."

    def go_home(self) -> str:
        if problem := self.check_connected():
            return problem
        self.shell("input keyevent KEYCODE_WAKEUP")
        self.shell("input keyevent KEYCODE_HOME")
        return "Phone is on its home screen."

    def lock(self) -> str:
        if problem := self.check_connected():
            return problem
        self.shell("input keyevent KEYCODE_SLEEP")
        return "Phone screen off and locked."

    def show(self) -> str:
        """Mirror and control the phone in a window on the Mac (scrcpy)."""
        if problem := self.check_connected():
            return problem
        scrcpy = find_tool("scrcpy")
        if scrcpy is None:
            return "Error: scrcpy isn't installed (brew install scrcpy)."
        env = dict(os.environ, ADB=self.adb)
        cmd = [scrcpy, "--window-title=Phone", "--stay-awake"] + (["--serial", self.serial] if self.serial else [])
        subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return "Your phone is now showing in a window on the Mac. If it's locked, it'll still need unlocking."

    def tools(self) -> list[Tool]:
        return [
            Tool(
                "unlock_phone",
                "Unlock the user's Android phone (wake it and enter their saved PIN).",
                params(),
                self.unlock,
                confirm=lambda: "Unlock your phone?",
            ),
            Tool("lock_phone", "Turn the user's phone screen off and lock it.", params(), self.lock),
            Tool(
                "open_phone_app",
                "Open an app on the user's phone. If it's locked, unlock_phone first.",
                params(
                    ["app"],
                    app={
                        "type": "string",
                        "description": "App name as the user said it, or its Android package if you know it "
                        "(e.g. 'com.google.android.gm' for Gmail)",
                    },
                ),
                self.open_app,
            ),
            Tool("phone_home", "Go to the phone's home screen.", params(), self.go_home),
            Tool(
                "show_phone",
                "Show and control the user's phone in a window on the Mac (screen mirroring).",
                params(),
                self.show,
            ),
        ]
