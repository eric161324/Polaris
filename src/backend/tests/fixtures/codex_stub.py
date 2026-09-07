#!/usr/bin/env python3
"""Offline CLI protocol fixture; never invokes a model or reads real credentials."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(os.environ["CODEX_HOME"])
scenario = json.loads((root / "scenario.json").read_text())
if "status" in sys.argv:
    print(scenario.get("login", "Logged in using ChatGPT"), file=sys.stderr)
    sys.exit(scenario.get("exit_code", 0))

prompt = sys.stdin.read()
(root / "capture.json").write_text(
    json.dumps(
        {
            "args": sys.argv[1:],
            "prompt": prompt,
            "cwd": os.getcwd(),
            "env": dict(os.environ),
            "images": [Path(arg).read_bytes().hex() for arg in sys.argv if arg.endswith(".png")],
        },
        ensure_ascii=False,
    )
)
mode = scenario.get("mode", "success")
if mode == "sleep":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (root / "pids.json").write_text(json.dumps([os.getpid(), child.pid]))
    time.sleep(60)
elif mode == "malformed":
    print("not-json secret-fixture-value")
elif mode == "error":
    print(json.dumps({"type": "turn.failed", "error": {"message": scenario["error"]}}))
elif mode == "native_tool":
    print(json.dumps({"type": "item.started", "item": {"type": "command_execution"}}))
else:
    print(json.dumps({"type": "thread.started", "thread_id": "offline-thread"}))
    print(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "agent_message",
                    "text": scenario.get("text", "你好，Polaris。"),
                },
            },
            ensure_ascii=False,
        )
    )
    if mode != "incomplete":
        print(
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 42,
                        "output_tokens": 9,
                        "cached_input_tokens": 12,
                    },
                }
            )
        )
sys.exit(scenario.get("exit_code", 0))
