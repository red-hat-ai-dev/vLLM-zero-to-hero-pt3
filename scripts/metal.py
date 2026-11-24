"""Manage a detached vLLM process using its checkout, PID and process start time."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid

CHECKOUT = str(Path(__file__).resolve().parents[1])


def process_start(pid):
    if type(pid) is not int or not 1 < pid < 2**31:
        raise ValueError("Invalid Metal PID")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart=", "-o", "stat="],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C", "TZ": "UTC"},
    )
    if result.returncode == 1 and not result.stderr.strip():
        return None
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(f"Cannot verify the start time of PID {pid}")
    started, status = result.stdout.strip().rsplit(maxsplit=1)
    return None if status.startswith("Z") else started


def read_live(record):
    try:
        st = record.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o022:
        raise ValueError(f"Untrusted Metal ownership record at {record}")
    info = json.loads(record.read_text())
    if isinstance(info, int):
        if info > 1 and process_start(info):
            raise ValueError(
                f"{record} is a live legacy PID. Verify it manually before removing the file."
            )
        return None
    if not isinstance(info, dict) or not all(
        info.get(k)
        for k in (
            "checkout",
            "pid",
            "started",
            "token",
            "mode",
            "port",
            "model",
            "command",
        )
    ):
        raise ValueError(
            f"Incomplete Metal ownership record at {record}; verify it manually."
        )
    if process_start(info["pid"]) != info["started"]:
        return None
    if info["checkout"] != CHECKOUT:
        raise ValueError("This Metal server belongs to a different lesson checkout")
    return info


def stop(info):
    for sig, timeout in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
        # Repeat the identity check before escalation in case the PID was reused.
        if process_start(info["pid"]) != info["started"]:
            return
        try:
            os.kill(info["pid"], sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process_start(info["pid"]) != info["started"]:
                return
            time.sleep(0.2)
    raise RuntimeError("Could not stop the recorded Metal process")


def launch(state, record, mode, port, command):
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", int(port))) == 0:
            raise ValueError(f"Port {port} is already in use.")
    log = state / "metal.log"
    with os.fdopen(
        os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600), "w"
    ) as output:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        started = process_start(child.pid)
        if not started or child.poll() is not None:
            raise RuntimeError(f"Metal failed to start. See {log}")
        info = dict(
            checkout=CHECKOUT,
            pid=child.pid,
            started=started,
            token=uuid.uuid4().hex,
            mode=mode,
            port=port,
            model=command[2],
            command=command,
        )
        with tempfile.NamedTemporaryFile(dir=state, mode="w", delete=False) as temp:
            try:
                json.dump(info, temp)
                temp.flush()
                os.replace(temp.name, record)
            finally:
                Path(temp.name).unlink(missing_ok=True)
        return info["token"]
    except BaseException:
        # This direct child has not been reaped by any other process.
        if child.poll() is None:
            child.kill()
            child.wait()
        raise


def manage(options):
    state = options.state
    record = state / "metal.pid"
    info = read_live(record)
    if options.action == "start":
        mode, port, *command = options.args
        command[0] = os.path.abspath(command[0])
        if Path(command[2]).is_dir():
            command[2] = str(Path(command[2]).resolve())
        if info:
            if (info["mode"], info["port"], info["command"]) != (mode, port, command):
                raise ValueError(
                    "A different Metal configuration is active. Run ./stop.sh first."
                )
            print("existing", info["token"])
        else:
            print("started", launch(state, record, mode, port, command))
    elif options.action == "status":
        if not info or (options.args and options.args != [info["token"]]):
            return 1
        print(json.dumps(info))
    else:
        if options.args and (not info or options.args != [info["token"]]):
            return 0  # An older startup must never cancel a newer launch.
        if info:
            stop(info)
        record.unlink(missing_ok=True)
        if options.action == "cleanup":
            (state / "metal.log").unlink(missing_ok=True)
            if not any(state.iterdir()):
                state.rmdir()
        print("vLLM Metal stopped." if info else "vLLM Metal is already stopped.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "stop", "cleanup"])
    parser.add_argument("state", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    # A waiting command may hold the old directory inode after cleanup.
    # Reopen it if another command removes or replaces the directory.
    while True:
        if options.action == "start":
            options.state.mkdir(mode=0o700, exist_ok=True)
        try:
            fd = os.open(options.state, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError:
            if options.action == "start":
                continue
            return 1 if options.action == "status" else 0
        try:
            st = os.fstat(fd)
            if st.st_uid != os.getuid() or st.st_mode & 0o022:
                raise ValueError(
                    f"Metal state must be owned and writable only by you: {options.state}"
                )
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                if not os.path.samestat(st, options.state.lstat()):
                    continue
            except FileNotFoundError:
                continue
            return manage(options)
        finally:
            os.close(fd)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        sys.exit(f"Metal: {exc}")
