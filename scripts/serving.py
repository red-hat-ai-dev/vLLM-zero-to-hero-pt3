"""Describe the server answering the client, including verified launcher details."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen


def command_json(args):
    """Optional launcher details are unavailable for external or stopped servers."""
    try:
        result = subprocess.run(args, capture_output=True, timeout=3, check=True)
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def container_details(container, port, alias, root):
    config = container.get("Config") or {}
    labels = config.get("Labels") or {}
    prefix = "io.github.red-hat-ai-dev.vllm-zero-to-hero-pt3"
    mode = labels.get(f"{prefix}.mode")
    args = config.get("Cmd") or []
    bindings = (container.get("NetworkSettings") or {}).get("Ports") or {}
    ports = bindings.get("8000/tcp") or []
    if (
        labels.get(f"{prefix}.managed") != "true"
        or mode not in ("baseline", "optimized")
        or not (container.get("State") or {}).get("Running")
        or not any(
            str(p.get("HostPort")) == str(port) and p.get("HostIp") == "127.0.0.1"
            for p in ports
        )
        or not args
        or args[0] != root
        or ("--served-model-name", alias) not in zip(args, args[1:])
    ):
        return None
    checkpoint = next(
        (
            mount["Source"]
            for mount in container.get("Mounts", [])
            if mount.get("Destination") == root and mount.get("Source")
        ),
        root,
    )
    return {
        "checkpoint": checkpoint,
        "mode": mode,
        "backend": labels.get(f"{prefix}.backend"),
        "image_id": container.get("Image"),
        "instance": container.get("Id"),
    }


def serving_identity(port, alias):
    with urlopen(f"http://127.0.0.1:{int(port)}/v1/models", timeout=5) as response:
        cards = json.load(response)["data"]
    card = next(
        (card for card in cards if isinstance(card, dict) and card.get("id") == alias),
        None,
    )
    if card is None:
        raise ValueError(f"Expected model alias {alias} is not being served")
    root = card.get("root")
    if root is not None and not isinstance(root, str):
        raise ValueError("The model card has an invalid checkpoint root")
    identity = {
        "alias": alias,
        "root": root,
        # These are launcher mount points. The same destination can contain
        # different checkpoints on different runs.
        "checkpoint": None
        if root in ("/models/baseline", "/models/optimized")
        else root,
        "max_model_len": card.get("max_model_len"),
    }
    if not root:
        return identity
    if sys.platform == "darwin":
        state = Path(os.environ.get("TMPDIR") or "/tmp") / "vllm-zero-to-hero-pt3"
        supervisor = Path(__file__).with_name("metal.py")
        status = command_json([sys.executable, str(supervisor), "status", str(state)])
        if (
            status
            and str(status.get("port")) == str(port)
            and status.get("model") == root
            and status.get("token")
        ):
            identity.update(
                mode=status["mode"],
                backend="metal",
                instance=hashlib.sha256(status["token"].encode()).hexdigest(),
            )
    elif sys.platform.startswith("linux"):
        engine = os.environ.get("ENGINE")
        for candidate in [engine] if engine else ["podman", "docker"]:
            containers = command_json(
                [candidate, "container", "inspect", "vllm-zero-to-hero-pt3"]
            )
            if isinstance(containers, list) and len(containers) == 1:
                details = container_details(containers[0], port, alias, root)
                if details:
                    identity.update(details)
                    break
    return identity


if __name__ == "__main__":
    try:
        print(json.dumps(serving_identity(*sys.argv[1:])))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        sys.exit(f"Server check failed: {exc}. Start ./run.sh first.")
