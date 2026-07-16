"""T42 取消钩子：acquire_model 装钩后，前向中途取消应立即抛 CancelledError。

覆盖四类模型形状的解析（nn.Module 本体 / mms 元组 / qwen 包装器 / 非 torch 对象）、
ScriptModule 跳过，以及 _align_one_qwen3 不再吞取消异常。全部离线。
"""
import pytest
import torch
import torch.nn as nn

from writansub.bridge import CancelledError, _resolve_torch_module


def _acquire(reg, name, obj):
    """经 acquire_model 注册（触发装钩），teardown 由 registry fixture 复位控制位。"""
    handle = reg.acquire_model(name, "cpu", lambda: obj)
    return reg.get_model(handle)


def test_module_forward_raises_on_cancel(registry):
    model = _acquire(registry, "hooktest:seq", nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)))
    x = torch.zeros(1, 4)
    model(x)  # 未取消：正常前向
    registry.cancelled = True
    with pytest.raises(CancelledError):
        model(x)
    registry.reset_controls()
    model(x)  # 复位后恢复可用


def test_tuple_resolution_mms_shape(registry):
    bundle = (nn.Linear(4, 4), "tokenizer", "aligner")
    model = _acquire(registry, "hooktest:tuple", bundle)[0]
    registry.cancelled = True
    with pytest.raises(CancelledError):
        model(torch.zeros(1, 4))


def test_wrapper_resolution_qwen_shape(registry):
    class Wrapper:
        def __init__(self):
            self.model = nn.Linear(4, 4)

    wrapper = _acquire(registry, "hooktest:wrapper", Wrapper())
    registry.cancelled = True
    with pytest.raises(CancelledError):
        wrapper.model(torch.zeros(1, 4))


def test_non_torch_object_skipped(registry):
    class FakeWhisper:
        model = object()  # CTranslate2 形状：.model 存在但非 nn.Module

    obj = _acquire(registry, "hooktest:ct2", FakeWhisper())
    assert isinstance(obj, FakeWhisper)  # 装钩静默跳过，不炸


def test_scriptmodule_not_hooked(registry):
    scripted = torch.jit.script(nn.Linear(4, 4))
    assert _resolve_torch_module(scripted) is None
    model = _acquire(registry, "hooktest:jit", scripted)
    registry.cancelled = True
    model(torch.zeros(1, 4))  # 无钩子：取消态下前向照常完成


def test_align_one_qwen3_propagates_cancel():
    from writansub.align.core import _align_one_qwen3
    from writansub.types import Sub

    class CancellingQwen:
        def align(self, **kwargs):
            raise CancelledError("任务已取消")

    sub = Sub(index=1, start=0.0, end=1.0, text="テスト")
    with pytest.raises(CancelledError):
        _align_one_qwen3(torch.zeros(1, 1600), sub, CancellingQwen(), 16000, "ja")
