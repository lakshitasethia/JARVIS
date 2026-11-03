"""Phone unlock safety rules, tested against a fake phone (no real adb or device needed)."""

import pytest

import jarvis.tools.phone as phone_mod
from jarvis.tools.phone import AndroidPhone, parse_awake, parse_keyguard_showing


class FakePhone(AndroidPhone):
    """Simulates a phone's lock screen. `correct_pin` unlocks it when typed at the lock screen."""

    def __init__(self, locked=True, correct_pin="1234", devices=("R5CY12345",), keyguard_readable=True):
        super().__init__(adb_path="/bin/echo", clock=lambda: self.now, sleep=lambda s: None)
        self.now = 1000.0
        self.locked = locked
        self.correct_pin = correct_pin
        self.devices = list(devices)
        self.keyguard_readable = keyguard_readable
        self.commands: list[str] = []

    def connected_devices(self):
        return self.devices

    def shell(self, command, timeout=10):
        self.commands.append(command)
        if command.startswith("dumpsys"):
            return f"mKeyguardShowing={str(self.locked).lower()}" if self.keyguard_readable else "nothing useful"
        if command.startswith("input text "):
            typed = command.removeprefix("input text ").strip("'")
            if self.locked and typed == self.correct_pin:
                self.locked = False
        return ""

    def typed_pin(self) -> bool:
        return any(c.startswith("input text") for c in self.commands)


@pytest.fixture(autouse=True)
def mac_unlocked_with_pin(monkeypatch):
    monkeypatch.setattr(phone_mod, "mac_is_locked", lambda: False)
    monkeypatch.setattr(phone_mod, "read_pin_from_keychain", lambda: "1234")


def test_unlocks_with_the_saved_pin():
    p = FakePhone()
    assert p.unlock() == "Phone unlocked."
    assert not p.locked
    assert p.commands[0] == "input keyevent KEYCODE_WAKEUP"
    assert "wm dismiss-keyguard" in p.commands


def test_refuses_while_the_mac_is_locked(monkeypatch):
    monkeypatch.setattr(phone_mod, "mac_is_locked", lambda: True)
    p = FakePhone()
    assert "Mac is locked" in p.unlock()
    assert p.commands == [] and p.locked


def test_never_types_the_pin_when_already_unlocked():
    """Otherwise the PIN could be typed into whatever app is open (e.g. a chat)."""
    p = FakePhone(locked=False)
    assert "already unlocked" in p.unlock()
    assert not p.typed_pin()


def test_never_types_the_pin_when_lock_state_is_unknown():
    p = FakePhone(keyguard_readable=False)
    assert "couldn't tell" in p.unlock()
    assert not p.typed_pin()


def test_wrong_pin_is_not_retried_and_cools_down(monkeypatch):
    monkeypatch.setattr(phone_mod, "read_pin_from_keychain", lambda: "9999")
    p = FakePhone()
    out = p.unlock()
    assert "still locked" in out and p.locked
    assert sum(c.startswith("input text") for c in p.commands) == 1  # exactly one attempt
    before = len(p.commands)
    assert "waiting" in p.unlock()  # cooldown: nothing sent to the phone
    assert len(p.commands) == before
    p.now += 301
    p.unlock()
    assert len(p.commands) > before  # allowed again after 5 minutes


def test_missing_or_weird_pin(monkeypatch):
    monkeypatch.setattr(phone_mod, "read_pin_from_keychain", lambda: None)
    assert "no phone PIN saved" in FakePhone().unlock()
    monkeypatch.setattr(phone_mod, "read_pin_from_keychain", lambda: "12;rm -rf")
    p = FakePhone()
    assert "letters and digits" in p.unlock() and not p.typed_pin()


def test_phone_not_connected():
    assert "isn't connected" in FakePhone(devices=()).unlock()
    assert "several Android phones" in FakePhone(devices=("a", "b")).unlock()


def test_unlock_tool_asks_for_confirmation():
    tools = {t.name: t for t in FakePhone().tools()}
    assert tools["unlock_phone"].confirm() == "Unlock your phone?"
    assert tools["lock_phone"].confirm is None


def test_lock():
    p = FakePhone(locked=False)
    assert p.lock() == "Phone screen off and locked."
    assert p.commands == ["input keyevent KEYCODE_SLEEP"]


def test_parsers():
    assert parse_keyguard_showing("foo\n  mKeyguardShowing=true\n") is True
    assert parse_keyguard_showing("isKeyguardShowing=false") is False
    assert parse_keyguard_showing("nothing here") is None
    assert parse_awake("mWakefulness=Awake") is True
    assert parse_awake("mWakefulness=Asleep") is False


def test_mac_lock_detection_reads_ioreg(monkeypatch):
    class Result:
        def __init__(self, out):
            self.stdout = out

    monkeypatch.undo()  # use the real mac_is_locked
    monkeypatch.setattr(phone_mod.subprocess, "run", lambda *a, **k: Result('"IOConsoleLocked" = No'))
    assert phone_mod.mac_is_locked() is False
    monkeypatch.setattr(phone_mod.subprocess, "run", lambda *a, **k: Result('"IOConsoleLocked" = Yes'))
    assert phone_mod.mac_is_locked() is True
    monkeypatch.setattr(phone_mod.subprocess, "run", lambda *a, **k: Result("garbage"))
    assert phone_mod.mac_is_locked() is True  # can't tell → assume locked (safe)


REAL_S25_OUTPUT = """List of devices attached
192.168.29.101:37461   device product:pa1qxins model:SM_S931B device:pa1q transport_id:2
adb-RZCY11MVJHX-GqeGv1._adb-tls-connect._tcp device product:pa1qxins model:SM_S931B device:pa1q transport_id:1
emulator-5554          offline
"""


def test_parse_adb_devices_keeps_ready_ones():
    from jarvis.tools.phone import parse_adb_devices

    assert parse_adb_devices(REAL_S25_OUTPUT) == [
        "192.168.29.101:37461",
        "adb-RZCY11MVJHX-GqeGv1._adb-tls-connect._tcp",
    ]


def test_same_phone_twice_counts_once_and_prefers_stable_name(monkeypatch):
    """Exactly what happened with the user's S25: manual connect + auto-discovery."""

    class Result:
        def __init__(self, out):
            self.stdout = out
            self.returncode = 0
            self.stderr = ""

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[1:] == ["devices", "-l"]:
            return Result(REAL_S25_OUTPUT)
        if cmd[-2:] == ["getprop", "ro.serialno"]:
            return Result("RZCY11MVJHX\n")
        return Result("mKeyguardShowing=true")

    monkeypatch.setattr(phone_mod.subprocess, "run", fake_run)
    p = AndroidPhone(adb_path="/bin/echo")
    assert p.connected_devices() == ["adb-RZCY11MVJHX-GqeGv1._adb-tls-connect._tcp"]
    assert p.check_connected() is None
    p.shell("dumpsys window")
    assert calls[-1][:3] == ["/bin/echo", "-s", "adb-RZCY11MVJHX-GqeGv1._adb-tls-connect._tcp"]


def test_parse_mdns_connect_addresses():
    from jarvis.tools.phone import parse_mdns_connect_addresses

    out = (
        "List of discovered mdns services\n"
        "adb-RZCY11MVJHX-GqeGv1\t_adb-tls-connect._tcp\t192.168.29.101:37461\n"
        "adb-RZCY11MVJHX-GqeGv1\t_adb-tls-pairing._tcp\t192.168.29.101:35825\n"
    )
    assert parse_mdns_connect_addresses(out) == ["192.168.29.101:37461"]


def test_reconnects_when_the_phone_dropped_off():
    """Asleep phones often vanish from `adb devices`; JARVIS should revive the link before giving up."""
    p = FakePhone(devices=())
    reconnects = []

    def revive():
        reconnects.append(1)
        p.devices = ["192.168.29.101:37461"]

    p.reconnect = revive
    assert p.unlock() == "Phone unlocked."
    assert reconnects == [1]


# A slice of the user's real S25 app list.
REAL_APPS = [
    "com.whatsapp", "com.google.android.gm", "com.grofers.customerapp", "com.instagram.android",
    "com.instagram.basel", "com.google.android.youtube", "com.google.android.apps.youtube.music",
    "com.sec.android.app.camera", "com.android.settings", "com.anthropic.claude", "com.application.zomato",
    "com.application.zomato.district", "com.google.android.apps.messaging", "org.telegram.messenger",
    "com.facebook.orca", "com.csam.icici.bank.imobile", "com.idfcfirstbank.optimus", "com.spotify.music",
]


def test_match_app_on_real_package_names():
    from jarvis.tools.phone import match_app

    cases = {
        "WhatsApp": "com.whatsapp", "Gmail": "com.google.android.gm", "Blinkit": "com.grofers.customerapp",
        "Instagram": "com.instagram.android", "YouTube": "com.google.android.youtube",
        "youtube music": "com.google.android.apps.youtube.music", "Claude": "com.anthropic.claude",
        "Zomato": "com.application.zomato", "messages": "com.google.android.apps.messaging",
        "com.idfcfirstbank.optimus": "com.idfcfirstbank.optimus",  # model passed the package itself
    }
    for name, expected in cases.items():
        assert match_app(name, REAL_APPS)[0] == expected, name
    assert match_app("Flurboxinator", REAL_APPS) == (None, [])


def test_open_app_requires_unlocked_phone_and_launches():
    p = FakePhone(locked=True)
    p._apps, p._apps_at = REAL_APPS, p.now
    assert "locked" in p.open_app("WhatsApp")
    assert not any(c.startswith("am start") for c in p.commands)
    p.locked = False
    original_shell = p.shell
    p.shell = lambda c, timeout=10: "com.whatsapp/.Main" if c.startswith("cmd package resolve-activity") else original_shell(c)
    assert p.open_app("WhatsApp") == "Opened com.whatsapp on the phone."
    assert "am start -n com.whatsapp/.Main" in p.commands
