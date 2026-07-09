"""translate_subs：T07 回退后的现状行为 + DEFAULT_BATCH_SIZE 常量。

用假 openai 模块隔离网络；registry 用真实单例（fixture 负责复位）。
"""
import inspect
import re
import sys
import types as _types

import pytest

from writansub.translate.core import DEFAULT_BATCH_SIZE, translate_subs
from writansub.types import Sub
from writansub.bridge import CancelledError


def _install_fake_openai(monkeypatch, handler):
    """handler(prompt, call_no) -> 回复文本，或抛异常模拟 API 失败。"""
    calls: list[str] = []

    class _Completions:
        def create(self, model=None, messages=None, temperature=None):
            prompt = messages[-1]["content"]
            calls.append(prompt)
            content = handler(prompt, len(calls))
            msg = _types.SimpleNamespace(content=content)
            return _types.SimpleNamespace(
                choices=[_types.SimpleNamespace(message=msg)])

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class OpenAI:
        def __init__(self, base_url=None, api_key=None):
            self.chat = _Chat()

    fake = _types.ModuleType("openai")
    fake.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)
    return calls


def _echo_reply(prompt, _call_no):
    out = []
    for line in prompt.splitlines():
        m = re.match(r"(\d+): ", line)
        if m:
            out.append(f"{m.group(1)}: 译{m.group(1)}")
    return "\n".join(out)


def _make_subs(n):
    return [Sub(index=i, start=float(i), end=float(i) + 1, text=f"原文{i}")
            for i in range(1, n + 1)]


def test_all_batches_translated(monkeypatch, registry):
    calls = _install_fake_openai(monkeypatch, _echo_reply)
    subs = _make_subs(45)
    result = translate_subs(subs, "简体中文", "http://fake", "k", "m",
                            batch_size=20)
    assert result is subs
    assert len(calls) == 3  # 45 条 / 20 -> 3 批
    assert all(s.translated == f"译{s.index}" for s in subs)


def test_multiline_reply_appended(monkeypatch, registry):
    def reply(prompt, _n):
        return "1: 第一句\n续行内容\n2: 第二句"

    _install_fake_openai(monkeypatch, reply)
    subs = _make_subs(2)
    translate_subs(subs, "简体中文", "http://fake", "k", "m")
    assert subs[0].translated == "第一句 续行内容"
    assert subs[1].translated == "第二句"


def test_failed_batch_skipped_others_survive(monkeypatch, registry):
    def reply(prompt, call_no):
        if call_no == 2:
            raise RuntimeError("api down")
        return _echo_reply(prompt, call_no)

    _install_fake_openai(monkeypatch, reply)
    logs = []
    subs = _make_subs(30)
    translate_subs(subs, "简体中文", "http://fake", "k", "m",
                   batch_size=10, log_callback=logs.append)
    ok = [s for s in subs if s.translated]
    assert {s.index for s in ok} == set(range(1, 11)) | set(range(21, 31))
    assert any("翻译出错" in m for m in logs)
    assert any("1 个批次失败" in m for m in logs)


def test_cancel_propagates(monkeypatch, registry):
    """现状契约（T07 未修）：取消时 CancelledError 向调用方传播。"""
    def reply(prompt, _n):
        registry.cancelled = True  # 首批完成后请求取消
        return _echo_reply(prompt, _n)

    calls = _install_fake_openai(monkeypatch, reply)
    subs = _make_subs(25)
    with pytest.raises(CancelledError):
        translate_subs(subs, "简体中文", "http://fake", "k", "m",
                       batch_size=10)
    assert len(calls) == 1  # 第二批 checkpoint 处即抛出


def test_signature_default_is_constant():
    sig = inspect.signature(translate_subs)
    assert sig.parameters["batch_size"].default is DEFAULT_BATCH_SIZE
