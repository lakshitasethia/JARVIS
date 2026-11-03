import plistlib
import subprocess
import sys

import pytest

import jarvis.service as service


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the service at a temp folder so tests never touch ~/Applications or LaunchAgents."""
    app = tmp_path / "Applications" / "JARVIS.app"
    monkeypatch.setattr(service, "APP", app)
    monkeypatch.setattr(service, "EXECUTABLE", app / "Contents" / "MacOS" / "JARVIS")
    monkeypatch.setattr(service, "LAUNCH_AGENT", tmp_path / "LaunchAgents" / f"{service.LABEL}.plist")
    monkeypatch.setattr(service, "SERVICE_LOG", tmp_path / "logs" / "service.log")
    return tmp_path


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_build_app_compiles_signs_and_is_idempotent(sandbox):
    assert service.build_app() is True
    info = plistlib.loads((service.APP / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == service.LABEL
    assert info["LSUIElement"] is True
    assert "Hey Jarvis" in info["NSMicrophoneUsageDescription"]
    assert service.EXECUTABLE.exists()
    # validly (ad-hoc) signed
    assert subprocess.run(["codesign", "--verify", str(service.APP)], capture_output=True).returncode == 0
    # same inputs → not rebuilt (rebuilding would make macOS re-ask for mic permission)
    assert service.build_app() is False


def test_launch_agent_starts_at_login_and_restarts_only_on_crash(sandbox):
    service.write_launch_agent()
    agent = plistlib.loads(service.LAUNCH_AGENT.read_bytes())
    assert agent["Label"] == service.LABEL
    assert agent["ProgramArguments"] == [str(service.EXECUTABLE)]
    assert agent["RunAtLoad"] is True
    assert agent["KeepAlive"] == {"SuccessfulExit": False}
    assert agent["WorkingDirectory"] == str(service.PROJECT)
    assert agent["StandardOutPath"] == str(service.SERVICE_LOG)


def test_only_one_listener_at_a_time(tmp_path):
    lock = tmp_path / "listening.lock"
    code = (
        "import sys, time, pathlib; from jarvis.main import _hold_single_listener_lock as h; "
        f"h(pathlib.Path({str(lock)!r})); print('got it', flush=True); time.sleep(5)"
    )
    first = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert first.stdout.readline().strip() == "got it"
        second = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        assert second.returncode == 1
        assert "already listening" in second.stderr
    finally:
        first.kill()
