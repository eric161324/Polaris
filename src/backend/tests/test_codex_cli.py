"""Behavioral contract for the subprocess adapter, including cancellation and locks."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.llm.base import (
    ImageBlock,
    Message,
    StreamDone,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
)
from app.core.llm.codex_cli import CodexCLIError, CodexCLIProvider
from app.core.llm.tool_stream import ToolCallAccumulator

TOOL = {
    "name": "search",
    "description": "Search papers",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}


@pytest.fixture
def cli(tmp_path):
    home = tmp_path / "account"
    home.mkdir()
    stub = Path(__file__).parent / "fixtures" / "codex_stub.py"
    binary = tmp_path / "codex"
    binary.write_text(f"#!{sys.executable}\n" + stub.read_text().split("\n", 1)[1])
    binary.chmod(0o700)
    provider = CodexCLIProvider(binary=str(binary), home=str(home), timeout=5, queue_timeout=1)

    def scenario(**values):
        (home / "scenario.json").write_text(json.dumps(values, ensure_ascii=False))

    scenario()
    return SimpleNamespace(provider=provider, home=home, scenario=scenario)


async def test_complete_preserves_context_images_and_scrubs_environment(cli, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret-api-fixture")
    monkeypatch.setenv("POLARIS_SECRET_KEY", "secret-app-fixture")
    monkeypatch.setenv("CODEX_THREAD_ID", "parent-task-fixture")
    messages = [
        Message("system", "用中文回答。"),
        Message("assistant", [ToolUseBlock("call1", "search", {"query": "论文"})]),
        Message(
            "user",
            [ToolResultBlock("call1", '{"title":"论文一"}', images=(ImageBlock(b"image-a"),))],
        ),
        Message("user", [TextBlock("比较结果"), ImageBlock(b"image-b")]),
    ]
    result = await cli.provider.complete(
        messages,
        model="fixture-model",
        images=[b"image-c"],
        effort="low",
        temperature=0.2,
        max_tokens=123,
    )
    assert result.content == "你好，Polaris。"
    assert result.usage == {"prompt_tokens": 42, "completion_tokens": 9, "cached_input_tokens": 12}
    capture = json.loads((cli.home / "capture.json").read_text())
    assert '"tool_use_id": "call1"' in capture["prompt"]
    assert "用中文回答。" in capture["prompt"] and "123" in capture["prompt"]
    assert capture["images"] == [data.hex() for data in (b"image-a", b"image-b", b"image-c")]
    assert "secret-api-fixture" not in json.dumps(capture)
    assert "secret-app-fixture" not in json.dumps(capture)
    assert "CODEX_THREAD_ID" not in capture["env"]
    assert "temperature" not in " ".join(capture["args"])
    assert "--ephemeral" in capture["args"] and "read-only" in capture["args"]
    assert 'forced_login_method="chatgpt"' in capture["args"]
    assert not Path(capture["cwd"]).exists()


async def test_tool_events_round_trip_and_plain_stream(cli):
    cli.scenario(
        text=json.dumps(
            {
                "content": "我会检索。",
                "tool_calls": [
                    {"name": "search", "arguments": '{"query":"扩散模型"}'},
                    {"name": "search", "arguments": '{"query":"图像生成"}'},
                ],
            }
        )
    )
    events = [
        event
        async for event in cli.provider.stream_events(
            [Message("user", "查找论文")],
            model="fixture-model",
            tools=[TOOL],
        )
    ]
    assert events[0] == TextDelta("我会检索。")
    assert isinstance(events[-1], StreamDone)
    assert events[-1].finish_reason == "tool_use"
    accumulator = ToolCallAccumulator()
    for event in events:
        accumulator.feed(event)
    calls = [b for b in accumulator.finish() if isinstance(b, ToolUseBlock)]
    assert len(calls) == 2 and calls[0].id != calls[1].id
    cli.scenario(text="根据检索结果，可以比较这两类论文。")
    messages = [
        Message("assistant", list(accumulator.finish())),
        Message("user", [ToolResultBlock(call.id, "检索结果") for call in calls]),
    ]
    chunks = [chunk async for chunk in cli.provider.stream(messages, model="fixture-model")]
    assert chunks == ["根据检索结果，可以比较这两类论文。"]
    prompt = json.loads((cli.home / "capture.json").read_text())["prompt"]
    assert all(prompt.count(call.id) == 2 for call in calls)


@pytest.mark.parametrize(
    "call,choice",
    [
        ({"name": "unknown", "arguments": "{}"}, None),
        ({"name": "search", "arguments": "not-json"}, None),
        ({"name": "search", "arguments": '{"query":1}'}, None),
        ({"name": "search", "arguments": '{"query":"x"}'}, "none"),
    ],
)
async def test_invalid_tool_requests_never_escape(cli, call, choice):
    cli.scenario(text=json.dumps({"content": "", "tool_calls": [call]}))
    with pytest.raises(CodexCLIError, match="CODEX_OUTPUT_INVALID"):
        await cli.provider.complete(
            [Message("user", "x")], model="m", tools=[TOOL], tool_choice=choice
        )


async def test_tool_choice_none_accepts_final_answer(cli):
    cli.scenario(text=json.dumps({"content": "最终答复", "tool_calls": []}))
    result = await cli.provider.complete(
        [Message("user", "x")], model="m", tools=[TOOL], tool_choice="none"
    )
    assert result.content == "最终答复" and not result.tool_calls


@pytest.mark.parametrize(
    "scenario,code",
    [
        ({"mode": "malformed"}, "CODEX_OUTPUT_INVALID"),
        ({"mode": "incomplete"}, "CODEX_OUTPUT_INVALID"),
        ({"mode": "native_tool"}, "CODEX_UNEXPECTED_TOOL"),
        (
            {"mode": "error", "error": "401 unauthorized secret-fixture-value"},
            "CODEX_NOT_LOGGED_IN",
        ),
        ({"mode": "error", "error": "usage limit secret-fixture-value"}, "CODEX_RATE_LIMIT"),
        (
            {"mode": "error", "error": "model is not supported secret-fixture-value"},
            "CODEX_MODEL_UNAVAILABLE",
        ),
    ],
)
async def test_safe_actionable_errors(cli, scenario, code):
    cli.scenario(**scenario)
    with pytest.raises(CodexCLIError, match=code) as error:
        await cli.provider.complete([Message("user", "x")], model="m")
    assert "secret-fixture-value" not in str(error.value)


async def test_login_status_requires_chatgpt_and_is_redacted(cli):
    assert await cli.provider.login_status() == {"ok": True, "error": None}
    cli.scenario(login="Logged in using an API key: secret-fixture-value")
    result = await cli.provider.login_status()
    assert not result["ok"] and "secret-fixture-value" not in str(result)
    cli.provider.binary = "/nonexistent/codex"
    assert "CODEX_CLI_NOT_FOUND" in (await cli.provider.login_status())["error"]


def running(pid):
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().split()[2] != "Z"


@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_cancel_kill_children_release_slot_and_remove_files(cli, cancel):
    cli.scenario(mode="sleep")
    cli.provider.timeout = 0.5
    task = asyncio.create_task(cli.provider.complete([Message("user", "x")], model="m"))
    async with asyncio.timeout(3):
        while not (cli.home / "pids.json").exists():
            await asyncio.sleep(0.01)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(CodexCLIError, match="CODEX_TIMEOUT"):
            await task
    pids = json.loads((cli.home / "pids.json").read_text())
    async with asyncio.timeout(3):
        while any(running(pid) for pid in pids):
            await asyncio.sleep(0.01)
    capture = json.loads((cli.home / "capture.json").read_text())
    assert not Path(capture["cwd"]).exists()
    cli.scenario()
    assert (await cli.provider.complete([Message("user", "x")], model="m")).content


async def test_cross_process_lock_and_queue_cancellation(cli):
    holder = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import fcntl,sys,time; f=open(sys.argv[1],'a'); "
        "fcntl.flock(f,fcntl.LOCK_EX); print('ready',flush=True); time.sleep(60)",
        str(cli.home / "polaris.lock"),
        stdout=asyncio.subprocess.PIPE,
    )
    try:
        await holder.stdout.readline()
        cli.provider.queue_timeout = 0.15
        with pytest.raises(CodexCLIError, match="CODEX_QUEUE_TIMEOUT"):
            await cli.provider.complete([Message("user", "x")], model="m")
        task = asyncio.create_task(cli.provider.complete([Message("user", "x")], model="m"))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not (cli.home / "capture.json").exists()
    finally:
        holder.kill()
        await holder.wait()
    assert (await cli.provider.complete([Message("user", "x")], model="m")).content


async def test_capabilities_stay_separate(cli):
    with pytest.raises(NotImplementedError):
        await cli.provider.embed(["x"], model="m")
    with pytest.raises(NotImplementedError):
        await cli.provider.rerank("x", ["x"], model="m")


def test_codex_proxy_is_applied_only_to_child_environment(cli, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "codex_proxy_url", "http://proxy.example:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://original.example:3128")
    assert cli.provider._env()["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert os.environ["HTTPS_PROXY"] == "http://original.example:3128"


def test_network_errors_suggest_the_codex_proxy_setting():
    from app.core.llm.codex_cli import _cli_error

    error = _cli_error("Reconnecting... 2/5 (request timed out)")
    assert "CODEX_NETWORK_ERROR" in str(error)
    assert "POLARIS_CODEX_PROXY_URL" in str(error)
